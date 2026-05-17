"""
finetune_pytorch.py — Fine-Tuning LoRA per Whisper con PyTorch su GPU Cloud.

Porting del pipeline MLX-native (07_finetune_mlx.py) a PyTorch + HuggingFace
per esecuzione su GPU NVIDIA (RunPod L4 24GB).

Features:
- Caricamento dati .npz preprocessati (compatibili con pipeline MLX)
- LoRA via PEFT (target: q_proj, v_proj)
- Training con Seq2SeqTrainer + fp16 + gradient checkpointing
- Evaluation con WER e Medical WER
- Early stopping configurabile
- Logging su Weights & Biases
- Supporto whisper-small / medium / large-v3

Usage:
    python finetune_pytorch.py --config training_config.yaml
    python finetune_pytorch.py --config training_config.yaml --model openai/whisper-medium
    python finetune_pytorch.py --config training_config.yaml --no-early-stopping
"""

import argparse
import os
import sys

# Assicura che i moduli locali (dataset.py, metrics.py) siano trovabili
# anche quando lo script è lanciato da una directory diversa.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
import wandb
from peft import LoraConfig, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from dataset import WhisperNpzDataset, DataCollatorForWhisperNpz
from metrics import load_medical_terms, make_compute_metrics


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Carica la configurazione YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def train(config: dict, cli_overrides: dict | None = None):
    """
    Esegue il fine-tuning LoRA di Whisper con PyTorch.

    Args:
        config: Configurazione dal file YAML.
        cli_overrides: Override da CLI per model, early stopping, etc.
    """
    cli_overrides = cli_overrides or {}

    # --- Parametri ---
    model_name = cli_overrides.get("model") or config["model"]["name"]
    language = config["model"]["language"]

    lora_rank = config["lora"]["rank"]
    lora_alpha = config["lora"]["alpha"]
    lora_dropout = config["lora"]["dropout"]
    target_modules = config["lora"]["target_modules"]

    batch_size = config["training"]["batch_size"]
    grad_accum = config["training"].get("gradient_accumulation_steps", 1)
    lr = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]
    warmup_steps = config["training"].get("warmup_steps", 100)
    weight_decay = config["training"]["weight_decay"]
    max_grad_norm = config["training"]["max_grad_norm"]
    use_fp16 = config["training"].get("fp16", True)
    use_grad_ckpt = config["training"].get("gradient_checkpointing", True)
    logging_steps = config["training"].get("logging_steps", 10)
    eval_steps = config["training"].get("eval_steps", 50)
    save_steps = config["training"].get("save_steps", 100)
    save_total_limit = config["training"].get("save_total_limit", 3)

    medical_terms_path = config["evaluation"]["medical_terms_path"]
    medical_weight = config["evaluation"]["medical_weight"]

    preprocessed_dir = config["data"]["preprocessed_dir"]
    output_dir = cli_overrides.get("output_dir") or config["data"]["output_dir"]
    seed = config["data"]["seed"]

    # Early stopping
    es_config = config.get("early_stopping", {})
    es_enabled = cli_overrides.get("es_enabled", es_config.get("enabled", True))
    es_patience = cli_overrides.get("es_patience") or es_config.get("patience", 3)
    es_metric = cli_overrides.get("es_metric") or es_config.get("metric", "eval_loss")

    # W&B
    wandb_config = config.get("wandb", {})
    wandb_project = wandb_config.get("project", "whisper-medical-finetuning")
    wandb_entity = wandb_config.get("entity")
    wandb_run_name = cli_overrides.get("run_name") or wandb_config.get("run_name")
    wandb_tags = wandb_config.get("tags", ["pytorch", "lora"])

    print("=" * 60)
    print("🚀 PyTorch LoRA Fine-Tuning — Whisper Cloud")
    print("=" * 60)

    # --- 1. Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- 2. Modello ---
    print(f"\n📦 Caricamento modello: {model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Disabilita cache per training (richiesto da gradient checkpointing)
    model.config.use_cache = False

    # Forza il decoder a generare in italiano (evita language detection in generate())
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    if use_grad_ckpt:
        model.gradient_checkpointing_enable()
        print("   ✅ Gradient checkpointing abilitato.")

    # --- 3. LoRA ---
    print(f"\n🔧 Applicazione LoRA (rank={lora_rank}, alpha={lora_alpha})")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Tokenizer ---
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name, language=language, task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task="transcribe"
    )

    # --- 5. Dataset ---
    train_dir = os.path.join(preprocessed_dir, "train")
    val_dir = os.path.join(preprocessed_dir, "val")

    if not os.path.exists(train_dir):
        print(f"❌ Directory training non trovata: {train_dir}")
        print("   Assicurati di aver copiato i dati preprocessati.")
        sys.exit(1)

    train_dataset = WhisperNpzDataset(train_dir)
    val_dataset = WhisperNpzDataset(val_dir) if os.path.exists(val_dir) else None

    data_collator = DataCollatorForWhisperNpz()

    print(f"\n📂 Dataset:")
    print(f"   Training:   {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset) if val_dataset else 0} samples")
    print(f"   Batch size: {batch_size} (effective: {batch_size * grad_accum})")

    # --- 6. Metriche ---
    medical_terms = set()
    if os.path.exists(medical_terms_path):
        medical_terms = load_medical_terms(medical_terms_path)
        print(f"   Glossario medico: {len(medical_terms)} termini")

    compute_metrics = make_compute_metrics(tokenizer, medical_terms, medical_weight)

    # --- 7. Training Arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        fp16=use_fp16,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=eval_steps if val_dataset else None,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        predict_with_generate=True,
        generation_max_length=448,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model=es_metric if val_dataset else None,
        greater_is_better=False if val_dataset else None,
        report_to="wandb",
        run_name=wandb_run_name,
        seed=seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # --- 8. Callbacks ---
    callbacks = []
    if es_enabled and val_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=es_patience))
        print(f"\n🛑 Early stopping: patience={es_patience}, metric={es_metric}")
    else:
        print("\n🛑 Early stopping: disabilitato")

    # --- 9. W&B Init ---
    os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity

    # --- 10. Trainer ---
    print(f"\n⚙️  Training args:")
    print(f"   Epochs: {num_epochs}")
    print(f"   LR: {lr}")
    print(f"   FP16: {use_fp16}")
    print(f"   Gradient checkpointing: {use_grad_ckpt}")
    print(f"   Grad accum steps: {grad_accum}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print(f"\n{'='*60}")
    print(f"🎯 Inizio training!")
    print(f"{'='*60}\n")

    # --- 11. Train ---
    trainer.train()

    # --- 12. Salvataggio finale ---
    final_dir = os.path.join(output_dir, "adapter_final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n💾 Adapter LoRA finali salvati in: {final_dir}")

    # Salva anche i risultati di eval
    if val_dataset:
        eval_results = trainer.evaluate()
        print(f"\n📊 Risultati finali:")
        for k, v in eval_results.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    print(f"\n{'='*60}")
    print(f"🎉 Training completato!")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-Tuning LoRA di Whisper con PyTorch su GPU cloud."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.yaml",
        help="Path al file di configurazione YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override del modello (es. openai/whisper-medium).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override della directory di output.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nome del run W&B (override YAML).",
    )

    # --- Early stopping CLI overrides ---
    es_group = parser.add_argument_group(
        "Early Stopping",
        "Override dei parametri di early stopping definiti nel YAML."
    )
    es_group.add_argument(
        "--es-patience",
        type=int,
        default=None,
        help="Epoche senza miglioramento prima dello stop.",
    )
    es_group.add_argument(
        "--es-metric",
        type=str,
        default=None,
        choices=["eval_loss", "eval_wer", "eval_medical_wer"],
        help="Metrica da monitorare per l'early stopping.",
    )
    es_group.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disabilita l'early stopping.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # Costruisci override
    cli_overrides = {}
    if args.model:
        cli_overrides["model"] = args.model
    if args.output_dir:
        cli_overrides["output_dir"] = args.output_dir
    if args.run_name:
        cli_overrides["run_name"] = args.run_name
    if args.no_early_stopping:
        cli_overrides["es_enabled"] = False
    if args.es_patience is not None:
        cli_overrides["es_patience"] = args.es_patience
    if args.es_metric is not None:
        cli_overrides["es_metric"] = args.es_metric

    train(config, cli_overrides=cli_overrides)


if __name__ == "__main__":
    main()
