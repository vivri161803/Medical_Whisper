"""
dataset.py — Custom PyTorch Dataset per file .npz preprocessati da Whisper MLX.

Carica i file .npz generati da 06_preprocess_mlx.py (formato nativo):
- log_mel: (3000, 80) float32  → trasposto a (80, 3000) per PyTorch Whisper
- labels: (448,) int32         → LongTensor
- text: stringa di riferimento
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WhisperNpzDataset(Dataset):
    """
    Dataset PyTorch che carica file .npz preprocessati per Whisper.

    Ogni file contiene:
    - log_mel: spettrogramma mel-log di shape (3000, 80) — formato MLX (time, mels)
    - labels: token target di shape (448,) con padding -100
    - text: testo di riferimento originale

    L'output traspone log_mel a (80, 3000) per compatibilità con
    WhisperForConditionalGeneration di HuggingFace (channels-first).
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory contenente i file .npz.
        """
        self.data_dir = Path(data_dir)
        self.files = sorted([
            self.data_dir / f
            for f in os.listdir(self.data_dir)
            if f.endswith(".npz")
        ])

        if not self.files:
            raise FileNotFoundError(
                f"Nessun file .npz trovato in: {data_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.files[idx], allow_pickle=True)

        # Trasponi mel da (3000, 80) → (80, 3000) per PyTorch Whisper
        log_mel = torch.from_numpy(data["log_mel"].T).float()

        # Labels come LongTensor
        labels = torch.from_numpy(data["labels"].copy()).long()

        return {
            "input_features": log_mel,
            "labels": labels,
        }


class DataCollatorForWhisperNpz:
    """
    Data collator per batch di campioni Whisper da file .npz.

    Gestisce:
    - Stack dei mel spectrogram (shape uniforme 80×3000)
    - Padding dei labels alla lunghezza massima del batch
    - Sostituzione di 0-padding con -100 per ignorare nella loss
    """

    def __call__(self, features: list[dict]) -> dict:
        # Stack mel — tutti hanno la stessa shape (80, 3000)
        input_features = torch.stack([f["input_features"] for f in features])

        # Labels — già padded a 448 nei .npz
        labels = torch.stack([f["labels"] for f in features])

        return {
            "input_features": input_features,
            "labels": labels,
        }

