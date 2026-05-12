#!/usr/bin/env python3
"""
03_finetune.py — Fine-Tuning di Whisper per Terminologia Medica

Allena Whisper Small sui segmenti audio preprocessati e filtrati,
utilizzando HuggingFace Seq2SeqTrainer con supporto MPS (Apple Silicon).

Uso:
    python scripts/03_finetune.py \
        --data_dir data/filtered \
        --config configs/training_config.yaml
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ---------------------------------------------------------------------------
# Configurazione di default
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "model_name": "openai/whisper-small",
    "language": "it",
    "task": "transcribe",
    "output_dir": "outputs/whisper-medical",
    "learning_rate": 1e-5,
    "warmup_steps": 500,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 10,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "wer",
    "greater_is_better": False,
    "logging_steps": 25,
    "fp16": False,
    "dataloader_num_workers": 0,
    "val_split": 0.1,
    "seed": 42,
    "max_input_length": 30.0,
}


def load_config(config_path: str | None) -> dict:
    """Carica configurazione da YAML, con fallback ai default."""
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)
        print(f"📋 Configurazione caricata da {config_path}")
    else:
        print("📋 Uso configurazione di default")
    return config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MedicalSpeechDataset(Dataset):
    """Dataset PyTorch per segmenti audio medici con trascrizioni."""

    def __init__(self, manifest: list[dict], processor: WhisperProcessor,
                 max_input_length: float = 30.0):
        self.manifest = manifest
        self.processor = processor
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        audio_path = entry["path"]
        text = entry["text"]

        # Carica audio
        audio_data, sr = sf.read(audio_path, dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample a 16kHz se necessario
        if sr != 16000:
            import torchaudio
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_data = torchaudio.functional.resample(
                audio_tensor, sr, 16000
            ).numpy()
            sr = 16000

        # Tronca a max_input_length
        max_samples = int(self.max_input_length * sr)
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]

        # Estrai features (log-Mel spectrogram)
        input_features = self.processor.feature_extractor(
            audio_data, sampling_rate=sr
        ).input_features[0]

        # Codifica labels (testo)
        labels = self.processor.tokenizer(text).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Collator personalizzato per Whisper:
    - Input features: già padded a 30s dal feature extractor
    - Labels: padding dinamico con -100 per ignorare nel calcolo della loss
    """
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Batch input features
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Batch labels con padding
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Maschera padding → -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Rimuovi BOS token se presente all'inizio di tutti i campioni
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------
def create_compute_metrics(processor: WhisperProcessor):
    """Crea la funzione di calcolo metriche (WER)."""
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Sostituisci -100 con pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decodifica
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        wer = 100 * wer_metric.compute(
            predictions=pred_str, references=label_str
        )
        return {"wer": wer}

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning di Whisper per terminologia medica"
    )
    parser.add_argument("--data_dir", default="data/filtered")
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    # Carica config
    config = load_config(args.config)

    # Carica manifest
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest_filtered.json"
    if not manifest_path.exists():
        print(f"❌ Manifest non trovato: {manifest_path}")
        print("   Esegui prima 02_filter_quality.py")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"📂 Caricati {len(manifest)} segmenti da {manifest_path}")

    if len(manifest) < 2:
        print("❌ Servono almeno 2 segmenti per train/val split")
        sys.exit(1)

    # Train/Val split
    np.random.seed(config["seed"])
    indices = np.random.permutation(len(manifest))
    val_size = max(1, int(len(manifest) * config["val_split"]))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_manifest = [manifest[i] for i in train_indices]
    val_manifest = [manifest[i] for i in val_indices]

    print(f"📊 Split: {len(train_manifest)} train / {len(val_manifest)} val")

    # Carica modello e processor
    model_name = config["model_name"]
    print(f"\n📦 Caricamento modello: {model_name}")

    processor = WhisperProcessor.from_pretrained(
        model_name, language=config["language"], task=config["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Configura il modello per la generazione
    model.generation_config.language = config["language"]
    model.generation_config.task = config["task"]
    model.generation_config.forced_decoder_ids = None

    # Seleziona device
    if torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    print(f"🖥️  Device: {device_str}")

    # Crea dataset
    print("📦 Preparazione dataset...")
    train_dataset = MedicalSpeechDataset(
        train_manifest, processor, config["max_input_length"]
    )
    val_dataset = MedicalSpeechDataset(
        val_manifest, processor, config["max_input_length"]
    )

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training arguments
    output_dir = config["output_dir"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        num_train_epochs=config["num_train_epochs"],
        eval_strategy=config["eval_strategy"],
        save_strategy=config["save_strategy"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        logging_steps=config["logging_steps"],
        fp16=config["fp16"],
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_num_workers=config["dataloader_num_workers"],
        seed=config["seed"],
        report_to=["tensorboard"],
        push_to_hub=False,
    )

    # Metriche
    compute_metrics = create_compute_metrics(processor)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Training
    print(f"\n🚀 Avvio fine-tuning...")
    print(f"   Epoche: {config['num_train_epochs']}")
    print(f"   Batch size effettivo: "
          f"{config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Output: {output_dir}\n")

    trainer.train()

    # Salva modello e processor
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))

    print(f"\n{'='*60}")
    print(f"✅ FINE-TUNING COMPLETATO")
    print(f"{'='*60}")
    print(f"  📁 Modello salvato in: {final_path}")
    print(f"  📊 Tensorboard: tensorboard --logdir {output_dir}")

    # Valutazione finale
    print("\n📊 Valutazione finale sul validation set...")
    metrics = trainer.evaluate()
    print(f"   WER: {metrics.get('eval_wer', 'N/A'):.2f}%")


if __name__ == "__main__":
    main()
