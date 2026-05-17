"""
05_baseline_benchmark.py — Benchmark Zero-Shot con mlx-whisper.

Trascrive un subset di validazione audio in modalità zero-shot (senza addestramento)
e registra il WER standard e Medical WER come baseline da abbattere.
"""

import argparse
import json
import os
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import mlx_whisper
import wandb

from scripts.metrics import compute_wer, compute_medical_wer, load_medical_terms


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, model: str = "mlx-community/whisper-small-mlx") -> str:
    """Trascrive un file audio usando mlx-whisper in modalità zero-shot."""
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model,
        language="it",
    )
    return result.get("text", "").strip()


def run_benchmark(
    manifest_path: str,
    medical_terms_path: str,
    output_dir: str,
    model: str = "mlx-community/whisper-small-mlx",
    medical_weight: float = 3.0,
    use_wandb: bool = True,
    num_samples: int = 0,
    seed: int = 42,
) -> dict:
    """
    Esegue il benchmark zero-shot sul manifest (o su un subset casuale).

    Args:
        manifest_path: Path al manifest JSON/JSONL con ground truth.
        medical_terms_path: Path al glossario dei termini medici.
        output_dir: Directory per il salvataggio del report.
        model: Nome/path del modello MLX Whisper.
        medical_weight: Peso per gli errori sui termini medici.
        use_wandb: Se True, logga i risultati su W&B.
        num_samples: Numero di campioni da processare (0 = tutti).
        seed: Seed per la riproducibilità del campionamento casuale.

    Returns:
        Report completo come dizionario.
    """
    # Carica manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            entries = json.loads(content)
        else:
            entries = [json.loads(line) for line in content.splitlines() if line.strip()]

    # Sampling casuale se richiesto
    total_available = len(entries)
    if num_samples and 0 < num_samples < total_available:
        random.seed(seed)
        entries = random.sample(entries, num_samples)
        print(f"🎲 Campionamento casuale: {num_samples}/{total_available} campioni (seed={seed}).")
    elif num_samples >= total_available:
        print(f"⚠️  num_samples ({num_samples}) >= totale ({total_available}): si processano tutti.")

    # Carica glossario medico
    medical_terms = load_medical_terms(medical_terms_path)
    print(f"📚 Glossario medico caricato: {len(medical_terms)} termini.")

    # Init W&B
    if use_wandb:
        wandb.init(
            project="whisper-medical-finetuning",
            tags=["baseline", "zero-shot"],
            config={
                "model": model,
                "manifest": manifest_path,
                "medical_weight": medical_weight,
                "num_samples": len(entries),
            },
        )

    file_results = []
    wer_scores = []
    medical_wer_scores = []

    from tqdm import tqdm

    for entry in tqdm(entries, desc="🎤 Trascrizione zero-shot"):
        audio_path = entry.get("audio_filepath", entry.get("audio_path", ""))
        reference = entry.get("text", entry.get("transcript", ""))
        entry_id = entry.get("id", "unknown")

        if not Path(audio_path).exists():
            print(f"  ⚠️  File non trovato: {audio_path}, skip.")
            continue

        try:
            hypothesis = transcribe_audio(audio_path, model=model)
        except Exception as e:
            print(f"  ❌ Errore trascrizione {entry_id}: {e}")
            continue

        # Calcola metriche
        try:
            wer_score = compute_wer(reference, hypothesis)
            med_wer_score = compute_medical_wer(
                reference, hypothesis, medical_terms, weight=medical_weight
            )
        except Exception as e:
            print(f"  ❌ Errore metriche {entry_id}: {e}")
            continue

        result = {
            "id": entry_id,
            "audio_path": audio_path,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(wer_score, 4),
            "medical_wer": round(med_wer_score, 4),
        }
        file_results.append(result)
        wer_scores.append(wer_score)
        medical_wer_scores.append(med_wer_score)

        # Log W&B per ogni file
        if use_wandb:
            wandb.log({
                "baseline/wer": wer_score,
                "baseline/medical_wer": med_wer_score,
            })

    # Aggregazione
    if wer_scores:
        aggregate = {
            "wer": {
                "mean": round(statistics.mean(wer_scores), 4),
                "median": round(statistics.median(wer_scores), 4),
                "min": round(min(wer_scores), 4),
                "max": round(max(wer_scores), 4),
                "stdev": round(statistics.stdev(wer_scores), 4) if len(wer_scores) > 1 else 0.0,
            },
            "medical_wer": {
                "mean": round(statistics.mean(medical_wer_scores), 4),
                "median": round(statistics.median(medical_wer_scores), 4),
                "min": round(min(medical_wer_scores), 4),
                "max": round(max(medical_wer_scores), 4),
                "stdev": round(statistics.stdev(medical_wer_scores), 4) if len(medical_wer_scores) > 1 else 0.0,
            },
        }
    else:
        aggregate = {"wer": {}, "medical_wer": {}}

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "manifest": manifest_path,
        "num_samples": len(file_results),
        "total_available": total_available,
        "sampling_seed": seed if (num_samples and 0 < num_samples < total_available) else None,
        "aggregate": aggregate,
        "file_results": file_results,
    }

    # Salva report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "baseline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Baseline Report:")
    if wer_scores:
        print(f"   WER medio:         {aggregate['wer']['mean']}")
        print(f"   Medical WER medio: {aggregate['medical_wer']['mean']}")
    print(f"   Report salvato in: {report_path}")

    # Log aggregati su W&B
    if use_wandb and wer_scores:
        wandb.summary["baseline/wer_mean"] = aggregate["wer"]["mean"]
        wandb.summary["baseline/medical_wer_mean"] = aggregate["medical_wer"]["mean"]
        wandb.finish()

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark zero-shot del modello whisper-small su dati di validazione."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path al manifest (JSON o JSONL) con ground truth.",
    )
    parser.add_argument(
        "--medical-terms",
        type=str,
        required=True,
        help="Path al file dei termini medici (un termine per riga).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory di output per il report (default: outputs).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-small-mlx",
        help="Nome o path del modello MLX Whisper.",
    )
    parser.add_argument(
        "--medical-weight",
        type=float,
        default=3.0,
        help="Peso penalizzante per errori su termini medici (default: 3.0).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disabilita il logging su Weights & Biases.",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=0,
        help="Numero di audio da campionare casualmente (0 = tutti, default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per la riproducibilità del campionamento casuale (default: 42).",
    )
    args = parser.parse_args()

    run_benchmark(
        manifest_path=args.manifest,
        medical_terms_path=args.medical_terms,
        output_dir=args.output_dir,
        model=args.model,
        medical_weight=args.medical_weight,
        use_wandb=not args.no_wandb,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
