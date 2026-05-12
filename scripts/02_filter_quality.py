#!/usr/bin/env python3
"""
02_filter_quality.py — Filtro Qualità Audio

Filtra i segmenti audio generati dallo Step 1, scartando quelli
con troppo silenzio, troppo rumore o durata inadeguata.

Usa Silero VAD per rilevare la percentuale di parlato e calcola
il rapporto segnale/rumore (SNR) per valutare la qualità audio.

Uso:
    python scripts/02_filter_quality.py \
        --input_dir data/chunks \
        --output_dir data/filtered
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Soglie di default
# ---------------------------------------------------------------------------
DEFAULT_MIN_SPEECH_RATIO = 0.3   # almeno 30% parlato
DEFAULT_MIN_SNR = 10.0           # almeno 10 dB
DEFAULT_MIN_DURATION = 1.0       # almeno 1 secondo
DEFAULT_MAX_DURATION = 30.0      # massimo 30 secondi
DEFAULT_MIN_SIMILARITY = 50.0    # almeno 50% similarity score


def load_silero_vad():
    """Carica il modello Silero VAD."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    return model, utils


def compute_speech_ratio(audio_path: str, vad_model, get_speech_ts) -> float:
    """
    Calcola la percentuale di audio che contiene parlato.
    Ritorna un valore tra 0.0 e 1.0.
    """
    audio_data, sr = sf.read(audio_path, dtype="float32")

    # Silero VAD vuole 16kHz mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    wav = torch.from_numpy(audio_data).float()
    if sr != 16000:
        # Resample semplice se necessario
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    total_samples = len(wav)
    if total_samples == 0:
        return 0.0

    speech_timestamps = get_speech_ts(wav, vad_model, sampling_rate=sr)

    speech_samples = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
    return speech_samples / total_samples


def compute_snr(audio_path: str, vad_model, get_speech_ts) -> float:
    """
    Stima il rapporto segnale/rumore (SNR) in dB.
    Usa le regioni di parlato (VAD) come segnale e le restanti come rumore.
    """
    audio_data, sr = sf.read(audio_path, dtype="float32")

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    wav = torch.from_numpy(audio_data).float()
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    speech_timestamps = get_speech_ts(wav, vad_model, sampling_rate=sr)
    wav_np = wav.numpy()

    # Raccogli campioni di parlato e di non-parlato
    speech_mask = np.zeros(len(wav_np), dtype=bool)
    for ts in speech_timestamps:
        speech_mask[ts["start"]:ts["end"]] = True

    speech_samples = wav_np[speech_mask]
    noise_samples = wav_np[~speech_mask]

    if len(speech_samples) == 0 or len(noise_samples) == 0:
        # Se non c'è parlato o non c'è rumore, non possiamo calcolare SNR
        return 0.0 if len(speech_samples) == 0 else 100.0

    # RMS energia
    speech_rms = np.sqrt(np.mean(speech_samples ** 2))
    noise_rms = np.sqrt(np.mean(noise_samples ** 2))

    if noise_rms < 1e-10:
        return 100.0  # Essenzialmente nessun rumore

    snr_db = 20 * np.log10(speech_rms / noise_rms)
    return float(snr_db)


def filter_segments(
    manifest: list[dict],
    output_dir: Path,
    vad_model,
    get_speech_ts,
    min_speech_ratio: float,
    min_snr: float,
    min_duration: float,
    max_duration: float,
    min_similarity: float,
) -> tuple[list[dict], dict]:
    """
    Filtra i segmenti applicando tutti i criteri di qualità.
    Ritorna (manifest_filtrato, statistiche).
    """
    stats = {
        "total": len(manifest),
        "passed": 0,
        "rejected_duration": 0,
        "rejected_speech_ratio": 0,
        "rejected_snr": 0,
        "rejected_similarity": 0,
    }

    filtered = []

    for entry in tqdm(manifest, desc="🔍 Filtering", unit="seg"):
        audio_path = entry["path"]
        duration = entry.get("duration", 0)
        similarity = entry.get("similarity_score", 0)

        # --- Filtro 1: Durata ---
        if duration < min_duration or duration > max_duration:
            stats["rejected_duration"] += 1
            continue

        # --- Filtro 2: Similarity score ---
        if similarity < min_similarity:
            stats["rejected_similarity"] += 1
            continue

        # --- Filtro 3: Speech ratio (VAD) ---
        try:
            speech_ratio = compute_speech_ratio(audio_path, vad_model, get_speech_ts)
        except Exception as e:
            print(f"  ⚠️  Errore VAD su {audio_path}: {e}")
            stats["rejected_speech_ratio"] += 1
            continue

        if speech_ratio < min_speech_ratio:
            stats["rejected_speech_ratio"] += 1
            continue

        # --- Filtro 4: SNR ---
        try:
            snr = compute_snr(audio_path, vad_model, get_speech_ts)
        except Exception as e:
            print(f"  ⚠️  Errore SNR su {audio_path}: {e}")
            stats["rejected_snr"] += 1
            continue

        if snr < min_snr:
            stats["rejected_snr"] += 1
            continue

        # --- Passato tutti i filtri ---
        dest_path = output_dir / Path(audio_path).name
        shutil.copy2(audio_path, dest_path)

        filtered_entry = entry.copy()
        filtered_entry["path"] = str(dest_path)
        filtered_entry["speech_ratio"] = round(speech_ratio, 3)
        filtered_entry["snr_db"] = round(snr, 1)
        filtered.append(filtered_entry)
        stats["passed"] += 1

    return filtered, stats


def main():
    parser = argparse.ArgumentParser(
        description="Filtro qualità audio: VAD + SNR + durata"
    )
    parser.add_argument("--input_dir", default="data/chunks")
    parser.add_argument("--output_dir", default="data/filtered")
    parser.add_argument("--min_speech_ratio", type=float, default=DEFAULT_MIN_SPEECH_RATIO)
    parser.add_argument("--min_snr", type=float, default=DEFAULT_MIN_SNR)
    parser.add_argument("--min_duration", type=float, default=DEFAULT_MIN_DURATION)
    parser.add_argument("--max_duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--min_similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Carica manifest
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ Manifest non trovato: {manifest_path}")
        print("   Esegui prima 01_chunk_and_align.py")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"📂 Caricati {len(manifest)} segmenti da {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Carica Silero VAD
    print("📦 Caricamento Silero VAD...")
    vad_model, utils = load_silero_vad()
    get_speech_ts = utils[0]  # get_speech_timestamps

    print(f"\n📋 Soglie di filtro:")
    print(f"   Speech ratio ≥ {args.min_speech_ratio}")
    print(f"   SNR ≥ {args.min_snr} dB")
    print(f"   Durata: {args.min_duration}s – {args.max_duration}s")
    print(f"   Similarity ≥ {args.min_similarity}%\n")

    # Filtra
    filtered, stats = filter_segments(
        manifest, output_dir, vad_model, get_speech_ts,
        args.min_speech_ratio, args.min_snr,
        args.min_duration, args.max_duration, args.min_similarity,
    )

    # Salva manifest filtrato
    filtered_manifest_path = output_dir / "manifest_filtered.json"
    with open(filtered_manifest_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    # Report
    print(f"\n{'='*60}")
    print(f"✅ FILTRO COMPLETATO")
    print(f"{'='*60}")
    print(f"  📊 Totale segmenti:         {stats['total']}")
    print(f"  ✅ Passati:                  {stats['passed']}")
    print(f"  ❌ Scartati (durata):        {stats['rejected_duration']}")
    print(f"  ❌ Scartati (speech ratio):  {stats['rejected_speech_ratio']}")
    print(f"  ❌ Scartati (SNR):           {stats['rejected_snr']}")
    print(f"  ❌ Scartati (similarity):    {stats['rejected_similarity']}")
    pct = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"  📈 Tasso accettazione:       {pct:.1f}%")
    print(f"\n  📄 Manifest: {filtered_manifest_path}")


if __name__ == "__main__":
    main()
