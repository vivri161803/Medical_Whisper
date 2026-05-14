"""
06_preprocess_mlx.py — Preprocessing Apple-Native per il fine-tuning.

Estrae feature log-Mel spettrogramma, tokenizza il testo, applica padding/masking,
esegue lo split train/val/test e salva tutto in file .npz compressi.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

# Costanti Whisper
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_N_MELS = 80
WHISPER_MAX_FRAMES = 3000  # 30 secondi × 100 frame/s


def extract_log_mel(
    audio_path: str,
    n_mels: int = WHISPER_N_MELS,
    sr: int = WHISPER_SAMPLE_RATE,
) -> np.ndarray:
    """
    Estrae le feature log-Mel spettrogramma usando la pipeline nativa di mlx-whisper.

    Questo garantisce che le feature siano nella stessa scala e formato
    usato dal modello Whisper MLX durante l'inferenza.

    Args:
        audio_path: Path al file audio WAV.
        n_mels: Numero di bande Mel (default 80 per Whisper).
        sr: Sample rate target (default 16000).

    Returns:
        Array numpy di shape (n_frames, n_mels) con le feature log-Mel
        nella scala nativa di mlx-whisper.
    """
    import mlx_whisper.audio as whisper_audio

    audio, orig_sr = sf.read(audio_path, dtype="float32")

    # Converti in mono se necessario
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample se necessario
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    # Usa la pipeline nativa di mlx-whisper (stessa scala del modello)
    audio_mx = mx.array(audio)
    mel = whisper_audio.log_mel_spectrogram(audio_mx, n_mels=n_mels)
    mx.eval(mel)

    return np.array(mel)  # shape: (n_frames, n_mels)


# ---------------------------------------------------------------------------
# Tokenizzazione (semplificata, basata su Whisper tokenizer)
# ---------------------------------------------------------------------------

def get_whisper_tokenizer():
    """
    Carica il tokenizer di Whisper via HuggingFace transformers.
    Usa il tokenizer multilingue di whisper-small con lingua italiana.
    """
    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small",
        language="it",
        task="transcribe",
    )
    return tokenizer


def tokenize_text(text: str, tokenizer) -> list[int]:
    """
    Tokenizza il testo usando il tokenizer di Whisper.

    Args:
        text: Testo da tokenizzare.
        tokenizer: Tokenizer di Whisper.

    Returns:
        Lista di token IDs.
    """
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
    else:
        tokens = tokenizer(text, return_tensors=None)["input_ids"]

    return list(tokens)


# ---------------------------------------------------------------------------
# Padding e Masking
# ---------------------------------------------------------------------------

def pad_or_trim_features(
    features: np.ndarray,
    max_frames: int = WHISPER_MAX_FRAMES,
) -> np.ndarray:
    """
    Allinea le feature a max_frames: tronca se più lunghe, padda con zeri se più corte.

    Args:
        features: Array di shape (n_frames, n_mels) — formato nativo mlx-whisper.
        max_frames: Numero massimo di frame (default 3000).

    Returns:
        Array di shape (n_frames, n_mels) con esattamente max_frames frame.
    """
    n_frames, n_mels = features.shape

    if n_frames > max_frames:
        return features[:max_frames, :]

    if n_frames < max_frames:
        padding = np.zeros((max_frames - n_frames, n_mels), dtype=features.dtype)
        return np.concatenate([features, padding], axis=0)

    return features


def pad_labels(
    labels: list[int],
    max_length: int,
    pad_token: int = -100,
) -> np.ndarray:
    """
    Padda i token label a max_length con pad_token (-100) per essere ignorati dalla loss.

    Args:
        labels: Lista di token IDs.
        max_length: Lunghezza massima.
        pad_token: Token di padding (default -100, ignorato nella cross-entropy).

    Returns:
        Array numpy di lunghezza max_length.
    """
    if len(labels) >= max_length:
        return np.array(labels[:max_length], dtype=np.int32)

    padded = labels + [pad_token] * (max_length - len(labels))
    return np.array(padded, dtype=np.int32)


# ---------------------------------------------------------------------------
# Split Train/Val/Test
# ---------------------------------------------------------------------------

def split_data(
    entries: list[dict],
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split deterministico del dataset in train/val/test.

    Args:
        entries: Lista di entry del manifest.
        split_ratios: Proporzioni (train, val, test). Devono sommare a 1.0.
        seed: Seed per il random shuffle.

    Returns:
        Tuple (train, val, test) di liste di entry.
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Le proporzioni devono sommare a 1.0"

    rng = np.random.default_rng(seed)
    indices = np.arange(len(entries))
    rng.shuffle(indices)

    n_total = len(entries)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [entries[i] for i in train_idx]
    val = [entries[i] for i in val_idx]
    test = [entries[i] for i in test_idx]

    return train, val, test


# ---------------------------------------------------------------------------
# Pipeline Completa
# ---------------------------------------------------------------------------

def preprocess_and_save(
    entries: list[dict],
    output_dir: str,
    split_name: str,
    tokenizer,
    max_label_length: int = 448,
):
    """
    Preprocessa una lista di entry e salva le feature in file .npz.

    Args:
        entries: Lista di entry (id, audio_path/audio_filepath, text/transcript).
        output_dir: Directory base di output.
        split_name: Nome dello split (train, val, test).
        tokenizer: Tokenizer di Whisper.
        max_label_length: Lunghezza massima dei label tokenizzati.
    """
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for entry in tqdm(entries, desc=f"  📦 {split_name}"):
        entry_id = entry.get("id", "unknown")
        audio_path = entry.get("audio_filepath", entry.get("audio_path", ""))
        text = entry.get("text", entry.get("transcript", ""))

        if not Path(audio_path).exists():
            print(f"  ⚠️  File non trovato: {audio_path}, skip.")
            continue

        try:
            # 1. Estrai log-Mel
            log_mel = extract_log_mel(audio_path)

            # 2. Padding/trimming delle feature
            log_mel_padded = pad_or_trim_features(log_mel)

            # 3. Tokenizza il testo
            tokens = tokenize_text(text, tokenizer)

            # 4. Padding dei label
            labels = pad_labels(tokens, max_label_length)

            # 5. Salva come .npz
            npz_path = os.path.join(split_dir, f"{entry_id}.npz")
            np.savez_compressed(
                npz_path,
                log_mel=log_mel_padded,
                labels=labels,
                text=text,
            )

        except Exception as e:
            print(f"  ❌ Errore per {entry_id}: {e}")
            continue


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing del dataset per il fine-tuning MLX-native."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path al manifest (JSON o JSONL).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/preprocessed",
        help="Directory di output per le feature preprocessate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per lo split deterministico.",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Proporzioni train/val/test (default: 0.8 0.1 0.1).",
    )
    parser.add_argument(
        "--max-label-length",
        type=int,
        default=448,
        help="Lunghezza massima dei label tokenizzati (default: 448).",
    )
    args = parser.parse_args()

    # Carica manifest
    print(f"📂 Caricamento manifest: {args.manifest}")
    with open(args.manifest, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            entries = json.loads(content)
        else:
            entries = [json.loads(line) for line in content.splitlines() if line.strip()]

    print(f"   {len(entries)} entry trovate.")

    # Split
    train, val, test = split_data(
        entries,
        split_ratios=tuple(args.split_ratios),
        seed=args.seed,
    )
    print(f"   Split: {len(train)} train / {len(val)} val / {len(test)} test")

    # Carica tokenizer
    print("🔤 Caricamento tokenizer Whisper...")
    tokenizer = get_whisper_tokenizer()

    # Preprocessa ogni split
    print("🔧 Preprocessing in corso...")
    for split_name, split_data_list in [("train", train), ("val", val), ("test", test)]:
        preprocess_and_save(
            entries=split_data_list,
            output_dir=args.output_dir,
            split_name=split_name,
            tokenizer=tokenizer,
            max_label_length=args.max_label_length,
        )

    # Salva i manifest degli split per riferimento
    for split_name, split_data_list in [("train", train), ("val", val), ("test", test)]:
        manifest_out = os.path.join(args.output_dir, split_name, "manifest.json")
        with open(manifest_out, "w", encoding="utf-8") as f:
            json.dump(split_data_list, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Preprocessing completato. Output in: {args.output_dir}")


if __name__ == "__main__":
    main()
