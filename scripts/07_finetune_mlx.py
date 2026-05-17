"""
07_finetune_mlx.py — Fine-Tuning LoRA nativo per MLX su Apple Silicon.

Training loop completo con:
- Caricamento pesi whisper-small tramite mlx_whisper.load_models.load_model()
- Iniezione LoRA su query e value (encoder + decoder attention)
- Training con mlx.nn.value_and_grad + mlx.optimizers.AdamW
- Evaluation intra-training con Medical WER
- Checkpointing incrementale degli adapter LoRA
- Logging su Weights & Biases
"""

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import wandb
from mlx.utils import tree_flatten, tree_unflatten

# Import locali
from scripts.config_loader import load_config
from scripts.metrics import compute_wer, compute_medical_wer, load_medical_terms


# ---------------------------------------------------------------------------
# LoRA Module
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Layer Lineare con adattamento LoRA (Low-Rank Adaptation).

    Il forward computa: y = W_frozen @ x + (alpha/rank) * B @ A @ x
    dove A e B sono le matrici low-rank trainabili.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        alpha: float = 64.0,
        dropout: float = 0.05,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Pesi frozen del layer originale (saranno caricati dopo)
        self.weight = mx.zeros((out_features, in_features))
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

        # Matrici LoRA trainabili
        self.lora_a = mx.random.normal((in_features, rank)) * (1.0 / math.sqrt(rank))
        self.lora_b = mx.zeros((rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        # Computazione frozen
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias

        # Computazione LoRA
        lora_x = x
        if self.dropout is not None:
            lora_x = self.dropout(lora_x)

        lora_out = (lora_x @ self.lora_a @ self.lora_b) * self.scale
        return y + lora_out


# ---------------------------------------------------------------------------
# LoRA Application
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 32,
    alpha: float = 64.0,
    dropout: float = 0.05,
) -> int:
    """
    Applica LoRA al modello Whisper MLX.

    Il modello mlx-whisper usa i nomi:
    - encoder.blocks.N.attn.query / .value
    - decoder.blocks.N.attn.query / .value
    - decoder.blocks.N.cross_attn.query / .value

    1. Congela tutti i parametri del modello.
    2. Per ogni blocco encoder/decoder, sostituisce i layer target con LoRALinear.
    3. I parametri LoRA (lora_a, lora_b) restano trainabili.

    Returns:
        Numero di layer sostituiti con LoRA.
    """
    model.freeze()
    replaced = 0

    def _inject_lora_in_attention(attn_module):
        nonlocal replaced
        for name in target_modules:
            if hasattr(attn_module, name):
                original = getattr(attn_module, name)
                if isinstance(original, nn.Linear):
                    lora_layer = LoRALinear(
                        in_features=original.weight.shape[1],
                        out_features=original.weight.shape[0],
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        bias=original.bias is not None,
                    )
                    # Copia i pesi frozen
                    lora_layer.weight = original.weight
                    if original.bias is not None:
                        lora_layer.bias = original.bias

                    # Congela tutto, scongela solo LoRA
                    lora_layer.freeze()
                    lora_layer.unfreeze(keys=["lora_a", "lora_b"])

                    setattr(attn_module, name, lora_layer)
                    replaced += 1

    # Encoder blocks
    if hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
        for block in model.encoder.blocks:
            if hasattr(block, "attn"):
                _inject_lora_in_attention(block.attn)

    # Decoder blocks
    if hasattr(model, "decoder") and hasattr(model.decoder, "blocks"):
        for block in model.decoder.blocks:
            if hasattr(block, "attn"):
                _inject_lora_in_attention(block.attn)
            if hasattr(block, "cross_attn"):
                _inject_lora_in_attention(block.cross_attn)

    return replaced


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Conta i parametri totali e trainabili del modello."""
    def _count(params):
        count = 0
        if isinstance(params, dict):
            for v in params.values():
                count += _count(v)
        elif isinstance(params, list):
            for v in params:
                count += _count(v)
        elif isinstance(params, mx.array):
            count += params.size
        return count

    total = _count(model.parameters())
    trainable = _count(model.trainable_parameters())
    return total, trainable


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def loss_fn(model: nn.Module, mel: mx.array, tokens: mx.array) -> mx.array:
    """
    Calcola la cross-entropy loss per il decoder di Whisper.

    Args:
        model: Modello Whisper con LoRA.
        mel: Feature log-Mel di shape (batch, n_mels, n_frames).
        tokens: Token target di shape (batch, seq_len).

    Returns:
        Loss scalare media.
    """
    # Il mel è già in formato (batch, n_frames, n_mels) — nativo mlx-whisper

    # Input decoder: tutti i token tranne l'ultimo
    decoder_input = tokens[:, :-1]
    # Target: tutti i token tranne il primo (shifted)
    target = tokens[:, 1:]

    # Forward pass: model(mel, tokens) → logits
    logits = model(mel, decoder_input)

    # Maschera per ignorare i token di padding (-100)
    mask = target != -100
    masked_target = mx.where(mask, target, mx.zeros_like(target))

    # Cross-entropy loss
    loss = nn.losses.cross_entropy(logits, masked_target, reduction="none")

    # Applica maschera e media
    masked_loss = loss * mask
    num_valid = mx.sum(mask)

    return mx.sum(masked_loss) / mx.maximum(num_valid, mx.array(1.0))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    val_files: list[str],
    batch_size: int,
) -> float:
    """
    Valuta il modello sui dati di validazione calcolando la loss media.

    Args:
        model: Modello Whisper con LoRA.
        val_files: Lista di file .npz di validazione.
        batch_size: Batch size per la valutazione.

    Returns:
        Loss media di validazione.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for i in range(0, len(val_files), batch_size):
        batch_files = val_files[i:i + batch_size]
        mel_list = []
        label_list = []

        for f in batch_files:
            data = np.load(f)
            mel_list.append(data["log_mel"])
            label_list.append(data["labels"])

        mel = mx.array(np.stack(mel_list))
        tokens = mx.array(np.stack(label_list))

        loss = loss_fn(model, mel, tokens)
        mx.eval(loss)
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def compute_epoch_wer(
    model: nn.Module,
    val_files: list[str],
    tokenizer,
    medical_terms: set[str],
    medical_weight: float = 3.0,
) -> dict:
    """
    Calcola WER e Medical WER sul validation set tramite trascrizione autoregressiva.

    Eseguito a fine epoca per monitorare la qualità effettiva del modello.

    Args:
        model: Modello Whisper con LoRA.
        val_files: Lista di file .npz di validazione.
        tokenizer: WhisperTokenizer per decodifica testo.
        medical_terms: Set di termini medici.
        medical_weight: Peso per errori su termini medici.

    Returns:
        Dizionario con metriche WER e Medical WER.
    """
    from mlx_whisper.decoding import DecodingOptions, decode as whisper_decode

    model.eval()

    wer_scores = []
    medical_wer_scores = []

    for f in val_files:
        data = np.load(f)
        mel = mx.array(data["log_mel"])  # (3000, 80) formato nativo mlx-whisper
        labels = data["labels"]

        # Decodifica i token di riferimento in testo
        # Rimuovi padding (-100) e token speciali Whisper (>= 50257)
        ref_tokens = [int(t) for t in labels if t != -100 and t < 50257]
        reference = tokenizer.decode(ref_tokens, skip_special_tokens=True).strip()

        if not reference:
            continue

        # Trascrizione autoregressiva con il modello LoRA
        try:
            # fp16=False: l'encoder produce features float32, il decoder deve accettarle
            options = DecodingOptions(
                language="it",
                task="transcribe",
                without_timestamps=True,
                fp16=False,
            )
            result = whisper_decode(model, mel, options)

            if isinstance(result, list):
                hypothesis = result[0].text.strip()
            else:
                hypothesis = result.text.strip()
        except Exception as e:
            print(f"    ⚠️  Decode fallito per {f}: {e}")
            hypothesis = ""

        # Calcola metriche
        try:
            wer_score = compute_wer(reference, hypothesis)
            wer_scores.append(wer_score)

            if medical_terms:
                med_wer = compute_medical_wer(
                    reference, hypothesis, medical_terms, weight=medical_weight
                )
                medical_wer_scores.append(med_wer)
        except Exception:
            continue

    model.train()

    results = {}
    if wer_scores:
        results["eval/wer"] = sum(wer_scores) / len(wer_scores)
    if medical_wer_scores:
        results["eval/medical_wer"] = sum(medical_wer_scores) / len(medical_wer_scores)

    return results


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(config: dict, es_overrides: dict | None = None):
    """
    Loop di training principale con calcolo reale di loss e gradienti.

    Args:
        config: Dizionario di configurazione dal file YAML.
        es_overrides: Override CLI per i parametri di early stopping.
    """
    # Parametri dalla config
    model_name = config["model"]["name"]
    lora_rank = config["lora"]["rank"]
    lora_alpha = config["lora"]["alpha"]
    lora_dropout = config["lora"]["dropout"]
    target_modules = config["lora"]["target_modules"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]
    weight_decay = config["training"]["weight_decay"]
    max_grad_norm = config["training"]["max_grad_norm"]
    eval_every = config["evaluation"]["eval_every_n_steps"]
    save_every = config["evaluation"]["save_every_n_steps"]
    medical_terms_path = config["evaluation"]["medical_terms_path"]
    medical_weight = config["evaluation"]["medical_weight"]
    preprocessed_dir = config["data"]["preprocessed_dir"]
    output_dir = config["data"]["output_dir"]

    # Early stopping config (YAML defaults + CLI override)
    es_config = config.get("early_stopping", {})
    if es_overrides:
        es_config.update({k: v for k, v in es_overrides.items() if v is not None})
    es_enabled = es_config.get("enabled", True)
    es_patience = es_config.get("patience", 3)
    es_metric = es_config.get("metric", "val_loss")
    es_min_delta = es_config.get("min_delta", 0.001)

    # W&B config
    wandb_config = config.get("wandb", {})
    wandb_project = wandb_config.get("project", "whisper-medical-finetuning")
    wandb_entity = wandb_config.get("entity")
    wandb_run_name = wandb_config.get("run_name")

    print("=" * 60)
    print("🚀 MLX LoRA Fine-Tuning — whisper-small")
    print("=" * 60)

    # Init W&B
    use_wandb = False
    try:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=config,
            tags=["finetuning", "lora", "mlx"],
        )
        use_wandb = True
        print("📊 W&B inizializzato.")
    except Exception as e:
        print(f"⚠️  W&B non disponibile: {e}. Logging solo su CLI.")

    # 1. Carica il modello Whisper come nn.Module
    print(f"\n📦 Caricamento modello: {model_name}")
    from mlx_whisper.load_models import load_model
    model = load_model(model_name, dtype=mx.float32)
    mx.eval(model.parameters())
    print(f"   ✅ Modello caricato.")

    # 2. Applica LoRA
    num_lora = apply_lora_to_model(
        model,
        target_modules=target_modules,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    total_params, trainable_params = count_parameters(model)
    pct = 100.0 * trainable_params / total_params if total_params > 0 else 0

    print(f"\n🔧 LoRA applicato:")
    print(f"   Layer sostituiti:     {num_lora}")
    print(f"   Parametri totali:     {total_params:,}")
    print(f"   Parametri trainabili: {trainable_params:,} ({pct:.2f}%)")
    print(f"   Rank: {lora_rank}, Alpha: {lora_alpha}, Dropout: {lora_dropout}")
    print(f"   Target: {target_modules}")

    # 3. Carica dati preprocessati
    train_dir = os.path.join(preprocessed_dir, "train")
    val_dir = os.path.join(preprocessed_dir, "val")

    if not os.path.exists(train_dir):
        print(f"❌ Directory training non trovata: {train_dir}")
        print("   Esegui prima 06_preprocess_mlx.py")
        return

    train_files = sorted([
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.endswith(".npz")
    ])

    val_files = sorted([
        os.path.join(val_dir, f)
        for f in os.listdir(val_dir)
        if f.endswith(".npz")
    ]) if os.path.exists(val_dir) else []

    print(f"\n📂 Dataset:")
    print(f"   Training:   {len(train_files)} samples")
    print(f"   Validation: {len(val_files)} samples")
    print(f"   Batch size: {batch_size}")

    # 4. Carica glossario medico (per eval)
    medical_terms = set()
    if os.path.exists(medical_terms_path):
        medical_terms = load_medical_terms(medical_terms_path)
        print(f"   Glossario medico: {len(medical_terms)} termini")

    # 5. Carica tokenizer per WER evaluation
    from transformers import WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="it", task="transcribe"
    )
    print(f"   Tokenizer caricato per WER evaluation.")

    # 6. Setup optimizer + loss_and_grad
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    print(f"\n⚙️  Optimizer: AdamW (lr={lr}, wd={weight_decay})")

    # 7. Setup output
    os.makedirs(output_dir, exist_ok=True)

    # 8. Training loop
    global_step = 0
    best_val_loss = float("inf")
    best_medical_wer = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    print(f"\n{'='*60}")
    print(f"🎯 Inizio training: {num_epochs} epoche")
    if es_enabled:
        print(f"🛑 Early stopping: patience={es_patience}, metric={es_metric}, min_delta={es_min_delta}")
    else:
        print(f"🛑 Early stopping: disabilitato")
    print(f"{'='*60}\n")

    model.train()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle dei file di training
        rng = np.random.default_rng(config["data"]["seed"] + epoch)
        shuffled_files = train_files.copy()
        rng.shuffle(shuffled_files)

        # Processa batch
        for i in range(0, len(shuffled_files), batch_size):
            batch_files = shuffled_files[i:i + batch_size]

            # Carica batch
            mel_list = []
            label_list = []
            for f in batch_files:
                data = np.load(f)
                mel_list.append(data["log_mel"])
                label_list.append(data["labels"])

            mel = mx.array(np.stack(mel_list))
            tokens = mx.array(np.stack(label_list))

            # Forward + backward pass reale
            loss, grads = loss_and_grad_fn(model, mel, tokens)

            # Gradient clipping
            grads_flat = tree_flatten(grads)
            grad_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in grads_flat if isinstance(g, mx.array)))
            if grad_norm.item() > max_grad_norm:
                scale = max_grad_norm / (grad_norm.item() + 1e-6)
                grads = tree_unflatten([
                    (k, g * scale if isinstance(g, mx.array) else g)
                    for k, g in grads_flat
                ])

            # Aggiorna i pesi LoRA
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1
            num_batches += 1

            # Log ogni 10 step
            if global_step % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(
                    f"  Epoch {epoch+1}/{num_epochs} | "
                    f"Step {global_step} | "
                    f"Loss: {loss_val:.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )

                if use_wandb:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/avg_loss": avg_loss,
                        "train/step": global_step,
                        "train/epoch": epoch + 1,
                        "train/grad_norm": grad_norm.item(),
                    })

            # Evaluation intra-training
            if val_files and global_step % eval_every == 0:
                val_loss = evaluate_model(model, val_files, batch_size)
                print(f"\n  📊 Eval step {global_step}: val_loss={val_loss:.4f}")

                if use_wandb:
                    wandb.log({
                        "eval/loss": val_loss,
                        "eval/step": global_step,
                    })

                # Salva il miglior modello
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(output_dir, "adapters_best.safetensors")
                    trainable_flat = dict(tree_flatten(model.trainable_parameters()))
                    mx.save_safetensors(best_path, trainable_flat)
                    print(f"  🏆 Nuovo miglior modello (val_loss={val_loss:.4f})")

                    if use_wandb:
                        wandb.summary["best_val_loss"] = val_loss
                        wandb.summary["best_step"] = global_step

            # Checkpointing periodico
            if global_step % save_every == 0:
                ckpt_path = os.path.join(output_dir, f"adapters_step{global_step}.safetensors")
                trainable_flat = dict(tree_flatten(model.trainable_parameters()))
                mx.save_safetensors(ckpt_path, trainable_flat)
                print(f"  💾 Checkpoint salvato: {ckpt_path}")

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(
            f"\n  ✅ Epoch {epoch+1}/{num_epochs} completata in {epoch_time:.1f}s | "
            f"Loss media: {avg_epoch_loss:.4f}"
        )

        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch_time_s": epoch_time,
            })

        # WER / Medical WER a fine epoca
        if val_files:
            print(f"\n  🔬 Calcolo WER/Medical WER (epoch {epoch+1})...")
            wer_results = compute_epoch_wer(
                model, val_files, tokenizer,
                medical_terms, medical_weight,
            )

            if "eval/wer" in wer_results:
                print(f"     WER:         {wer_results['eval/wer']:.4f}")
            if "eval/medical_wer" in wer_results:
                print(f"     Medical WER: {wer_results['eval/medical_wer']:.4f}")

                if wer_results["eval/medical_wer"] < best_medical_wer:
                    best_medical_wer = wer_results["eval/medical_wer"]
                    print(f"     🏆 Nuovo miglior Medical WER!")

            if use_wandb and wer_results:
                wandb.log({
                    **wer_results,
                    "eval/epoch": epoch + 1,
                })

        # --- Early Stopping Check ---
        if es_enabled and val_files:
            if es_metric == "medical_wer" and "eval/medical_wer" in wer_results:
                current_value = wer_results["eval/medical_wer"]
                best_value = best_medical_wer
            else:
                # Calcola val_loss di fine epoca come metrica di default
                end_epoch_val_loss = evaluate_model(model, val_files, batch_size)
                current_value = end_epoch_val_loss
                best_value = best_val_loss

            # Verifica miglioramento
            improvement = best_value - current_value
            if improvement > es_min_delta:
                epochs_without_improvement = 0
                print(f"  📈 Early stopping: metrica migliorata di {improvement:.4f} (contatore resettato).")
            else:
                epochs_without_improvement += 1
                print(
                    f"  📉 Early stopping: nessun miglioramento significativo "
                    f"({epochs_without_improvement}/{es_patience})."
                )

            if use_wandb:
                wandb.log({"early_stopping/epochs_without_improvement": epochs_without_improvement})

            if epochs_without_improvement >= es_patience:
                print(
                    f"\n  🛑 Early stopping attivato! Nessun miglioramento per "
                    f"{es_patience} epoche consecutive."
                )
                stopped_early = True
                break

    # Salvataggio finale
    final_path = os.path.join(output_dir, "adapters_final.safetensors")
    trainable_flat = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(final_path, trainable_flat)
    print(f"\n💾 Adapter finali salvati in: {final_path}")

    # Copia config nel output
    config_src = os.path.join(os.path.dirname(__file__), "..", "training_config.yaml")
    config_dst = os.path.join(output_dir, "training_config.yaml")
    if os.path.exists(config_src):
        shutil.copy2(config_src, config_dst)

    # Summary W&B
    if use_wandb:
        wandb.summary["final_train_loss"] = avg_epoch_loss
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["best_medical_wer"] = best_medical_wer if best_medical_wer < float("inf") else None
        wandb.summary["total_steps"] = global_step
        wandb.summary["lora_layers"] = num_lora
        wandb.summary["trainable_params"] = trainable_params
        wandb.summary["trainable_pct"] = pct
        wandb.summary["stopped_early"] = stopped_early
        wandb.summary["epochs_completed"] = epoch + 1
        wandb.finish()

    stop_reason = "early stopping" if stopped_early else "completamento epoche"
    print(f"\n{'='*60}")
    print(f"🎉 Training terminato ({stop_reason})!")
    print(f"   Epoche completate: {epoch + 1}/{num_epochs}")
    print(f"   Loss finale: {avg_epoch_loss:.4f}")
    print(f"   Miglior val_loss: {best_val_loss:.4f}")
    if best_medical_wer < float("inf"):
        print(f"   Miglior Medical WER: {best_medical_wer:.4f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-Tuning LoRA di whisper-small con MLX."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.yaml",
        help="Path al file di configurazione YAML.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path opzionale a un checkpoint precedente (adapters.safetensors).",
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
        help="Epoche senza miglioramento prima dello stop (override YAML).",
    )
    es_group.add_argument(
        "--es-metric",
        type=str,
        default=None,
        choices=["val_loss", "medical_wer"],
        help="Metrica da monitorare: 'val_loss' o 'medical_wer' (override YAML).",
    )
    es_group.add_argument(
        "--es-min-delta",
        type=float,
        default=None,
        help="Miglioramento minimo per considerare progresso (override YAML).",
    )
    es_group.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disabilita l'early stopping (ignora config YAML).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.resume:
        print(f"📂 Ripresa dal checkpoint: {args.resume}")

    # Costruisci override dict per early stopping
    es_overrides = {}
    if args.no_early_stopping:
        es_overrides["enabled"] = False
    if args.es_patience is not None:
        es_overrides["patience"] = args.es_patience
    if args.es_metric is not None:
        es_overrides["metric"] = args.es_metric
    if args.es_min_delta is not None:
        es_overrides["min_delta"] = args.es_min_delta

    train(config, es_overrides=es_overrides or None)


if __name__ == "__main__":
    main()
