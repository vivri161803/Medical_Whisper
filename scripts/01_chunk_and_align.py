#!/usr/bin/env python3
"""
01_chunk_and_align.py — Segmentazione Audio e Allineamento con Trascrizioni

Prende audio lunghi (>1h) e le relative trascrizioni manuali (.txt),
usa WhisperX per segmentare e allineare, poi sostituisce il testo
di ogni segmento con il ground truth della trascrizione manuale
tramite fuzzy matching.

Uso:
    python scripts/01_chunk_and_align.py \
        --input_dir data/raw \
        --output_dir data/chunks \
        --language it \
        --model_size small
"""

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
import whisperx
from rapidfuzz import fuzz
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE = "it"
DEFAULT_MODEL_SIZE = "small"
DEFAULT_MIN_SIMILARITY = 60
DEFAULT_BATCH_SIZE = 16
MIN_SEGMENT_DURATION = 1.0
MAX_SEGMENT_DURATION = 30.0
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma"}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_compute_type(device: str) -> str:
    if device == "cuda":
        return "float16"
    return "float32"


def find_audio_transcript_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Trova coppie (audio, trascrizione) con lo stesso nome base."""
    pairs = []
    txt_files = {f.stem: f for f in input_dir.iterdir() if f.suffix == ".txt"}
    for audio_file in sorted(input_dir.iterdir()):
        if audio_file.suffix.lower() in AUDIO_EXTENSIONS:
            if audio_file.stem in txt_files:
                pairs.append((audio_file, txt_files[audio_file.stem]))
            else:
                print(f"⚠️  Nessuna trascrizione per: {audio_file.name}")
    return pairs


def load_transcript(txt_path: Path) -> str:
    text = txt_path.read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text).strip()


def merge_short_segments(segments, min_dur):
    if not segments:
        return segments
    merged = [segments[0]]
    for seg in segments[1:]:
        if (seg["end"] - seg["start"]) < min_dur and merged:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = merged[-1]["text"] + " " + seg["text"]
        else:
            merged.append(seg)
    return merged


def split_long_segments(segments, max_dur):
    result = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur <= max_dur:
            result.append(seg)
        else:
            n = int(dur // max_dur) + 1
            words = seg["text"].split()
            wpn = max(1, len(words) // n)
            pd = dur / n
            for i in range(n):
                sw = i * wpn
                ew = sw + wpn if i < n - 1 else len(words)
                t = " ".join(words[sw:ew])
                if t.strip():
                    result.append({
                        "start": seg["start"] + i * pd,
                        "end": seg["start"] + (i + 1) * pd,
                        "text": t.strip(),
                    })
    return result


def fuzzy_align_segment(seg_text, full_transcript, min_sim):
    """Cerca la sottostringa più simile nella trascrizione manuale."""
    seg_text = seg_text.strip()
    if not seg_text:
        return seg_text, 0.0

    seg_len = len(seg_text)
    best_score = 0.0
    best_match = seg_text

    window_sizes = [
        int(seg_len * f) for f in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    ]

    for win in window_sizes:
        if win < 5 or win > len(full_transcript):
            continue
        step = max(1, win // 4)
        for start in range(0, len(full_transcript) - win + 1, step):
            candidate = full_transcript[start:start + win]
            score = fuzz.ratio(seg_text.lower(), candidate.lower())
            if score > best_score:
                best_score = score
                best_match = candidate

    if best_score >= min_sim:
        return best_match.strip(), best_score
    return seg_text, best_score


def process_audio_pair(audio_path, transcript_path, output_dir, model,
                       align_model, align_metadata, device, language,
                       min_similarity, batch_size):
    """Processa una coppia audio + trascrizione."""
    print(f"\n{'='*60}")
    print(f"📂 Processing: {audio_path.name}")
    print(f"{'='*60}")

    audio = whisperx.load_audio(str(audio_path))

    print("  🎤 Trascrizione con WhisperX...")
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    segments = result.get("segments", [])
    print(f"  ✅ {len(segments)} segmenti trascritti")

    print("  🔗 Forced alignment...")
    result = whisperx.align(
        segments, align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )
    segments = result.get("segments", [])

    segments = merge_short_segments(segments, MIN_SEGMENT_DURATION)
    segments = split_long_segments(segments, MAX_SEGMENT_DURATION)
    print(f"  📐 {len(segments)} segmenti dopo merge/split")

    full_transcript = load_transcript(transcript_path)
    print(f"  📝 Trascrizione caricata ({len(full_transcript)} chars)")

    print("  🔍 Fuzzy alignment...")
    manifest_entries = []
    sr = 16000

    for i, seg in enumerate(tqdm(segments, desc="  Allineamento", unit="seg")):
        whisperx_text = seg.get("text", "").strip()
        start_t, end_t = seg["start"], seg["end"]

        aligned_text, similarity = fuzzy_align_segment(
            whisperx_text, full_transcript, min_similarity
        )

        segment_audio = audio[int(start_t * sr):int(end_t * sr)]
        seg_fname = f"{audio_path.stem}_{i:04d}.wav"
        seg_path = output_dir / seg_fname
        sf.write(str(seg_path), segment_audio, sr)

        manifest_entries.append({
            "path": str(seg_path),
            "filename": seg_fname,
            "text": aligned_text,
            "whisperx_text": whisperx_text,
            "start": round(start_t, 3),
            "end": round(end_t, 3),
            "duration": round(end_t - start_t, 3),
            "source_file": audio_path.name,
            "similarity_score": round(similarity, 1),
        })

    if manifest_entries:
        avg = sum(e["similarity_score"] for e in manifest_entries) / len(manifest_entries)
        print(f"  📊 Similarità media: {avg:.1f}%")

    return manifest_entries


def main():
    parser = argparse.ArgumentParser(
        description="Segmentazione audio e allineamento con WhisperX"
    )
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/chunks")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--model_size", default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--min_similarity", type=int, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"❌ Cartella non trovata: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_audio_transcript_pairs(input_dir)
    if not pairs:
        print(f"❌ Nessuna coppia audio+txt in {input_dir}")
        sys.exit(1)

    print(f"🎯 Trovate {len(pairs)} coppie")
    for a, t in pairs:
        print(f"   • {a.name} ↔ {t.name}")

    device = get_device()
    compute_type = get_compute_type(device)
    print(f"\n🖥️  Device: {device} | Compute: {compute_type}")

    print(f"\n📦 Caricamento WhisperX ({args.model_size})...")
    model = whisperx.load_model(
        args.model_size, device, compute_type=compute_type, language=args.language
    )

    print("📦 Caricamento modello allineamento...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=args.language, device=device
    )

    all_entries = []
    for audio_path, txt_path in pairs:
        entries = process_audio_pair(
            audio_path, txt_path, output_dir, model, align_model,
            align_metadata, device, args.language, args.min_similarity,
            args.batch_size,
        )
        all_entries.extend(entries)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    del model, align_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"✅ COMPLETATO — {len(all_entries)} segmenti")
    print(f"   Manifest: {manifest_path}")
    if all_entries:
        avg_sim = sum(e["similarity_score"] for e in all_entries) / len(all_entries)
        total_min = sum(e["duration"] for e in all_entries) / 60
        print(f"   Similarità media: {avg_sim:.1f}%")
        print(f"   Durata totale: {total_min:.1f} min")


if __name__ == "__main__":
    main()
