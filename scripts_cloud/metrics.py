"""
metrics.py — Metriche di valutazione per il fine-tuning PyTorch cloud.

Port standalone del modulo scripts/metrics.py per funzionare senza dipendenze MLX.

Fornisce:
- compute_wer(): WER standard via jiwer.
- load_medical_terms(): caricamento glossario medico da file esterno.
- compute_medical_wer(): WER pesato con penalità su terminologia medica.
- make_compute_metrics(): factory per la funzione compute_metrics compatibile con Seq2SeqTrainer.
"""

from pathlib import Path

import jiwer
import numpy as np


# ---------------------------------------------------------------------------
# WER Standard
# ---------------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Calcola il Word Error Rate (WER) standard tra reference e hypothesis.

    Args:
        reference: Testo di riferimento (ground truth).
        hypothesis: Testo ipotizzato (trascrizione del modello).

    Returns:
        WER come float (0.0 = perfetto, 1.0 = 100% errori).
    """
    if not reference.strip():
        raise ValueError("Il testo di riferimento non può essere vuoto.")
    return jiwer.wer(reference, hypothesis)


# ---------------------------------------------------------------------------
# Glossario Medico
# ---------------------------------------------------------------------------

def load_medical_terms(path: str) -> set[str]:
    """
    Carica il glossario dei termini medici da file esterno.

    Il file deve contenere un termine per riga, case-insensitive.
    Le righe vuote e quelle che iniziano con '#' vengono ignorate.

    Args:
        path: Path al file dei termini medici.

    Returns:
        Set di termini medici in minuscolo.
    """
    terms_path = Path(path)
    if not terms_path.exists():
        raise FileNotFoundError(f"File glossario non trovato: {path}")

    terms: set[str] = set()
    with open(terms_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.add(line.lower())
    return terms


def _is_medical_token(token: str, medical_terms: set[str]) -> bool:
    """Verifica se un token è un termine medico."""
    return token.lower() in medical_terms


# ---------------------------------------------------------------------------
# Medical WER
# ---------------------------------------------------------------------------

def compute_medical_wer(
    reference: str,
    hypothesis: str,
    medical_terms: set[str],
    weight: float = 3.0,
) -> float:
    """
    Calcola il Medical WER pesato.

    Gli errori (sostituzioni, cancellazioni, inserzioni) su token presenti
    nel glossario medico sono moltiplicati per `weight`.

    Args:
        reference: Testo di riferimento (ground truth).
        hypothesis: Testo ipotizzato (trascrizione del modello).
        medical_terms: Set di termini medici (in minuscolo).
        weight: Moltiplicatore per errori su termini medici (default 3.0).

    Returns:
        Medical WER come float.
    """
    if not reference.strip():
        raise ValueError("Il testo di riferimento non può essere vuoto.")

    output = jiwer.process_words(reference, hypothesis)

    weighted_errors = 0.0
    weighted_total = 0.0

    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            ref_words = output.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
            for w in ref_words:
                w_weight = weight if _is_medical_token(w, medical_terms) else 1.0
                weighted_total += w_weight

        elif chunk.type == "substitute":
            ref_words = output.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
            for w in ref_words:
                w_weight = weight if _is_medical_token(w, medical_terms) else 1.0
                weighted_total += w_weight
                weighted_errors += w_weight

        elif chunk.type == "delete":
            ref_words = output.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
            for w in ref_words:
                w_weight = weight if _is_medical_token(w, medical_terms) else 1.0
                weighted_total += w_weight
                weighted_errors += w_weight

        elif chunk.type == "insert":
            hyp_words = output.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx]
            for w in hyp_words:
                w_weight = weight if _is_medical_token(w, medical_terms) else 1.0
                weighted_errors += w_weight

    if weighted_total == 0.0:
        return 0.0

    return weighted_errors / weighted_total


# ---------------------------------------------------------------------------
# Trainer-Compatible Compute Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(tokenizer, medical_terms: set[str], medical_weight: float = 3.0):
    """
    Factory che crea la funzione compute_metrics compatibile con Seq2SeqTrainer.

    Args:
        tokenizer: WhisperTokenizer per decodifica.
        medical_terms: Set di termini medici.
        medical_weight: Peso penalizzante per errori medici.

    Returns:
        Funzione compute_metrics(eval_preds) → dict.
    """

    def compute_metrics(eval_preds) -> dict:
        pred_ids, label_ids = eval_preds

        # Sostituisci -100 con pad_token_id per la decodifica
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)

        # Decodifica batch
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Calcola WER e Medical WER
        wer_scores = []
        medical_wer_scores = []

        for ref, hyp in zip(label_str, pred_str):
            ref = ref.strip()
            hyp = hyp.strip()

            if not ref:
                continue

            try:
                wer_scores.append(compute_wer(ref, hyp))
                if medical_terms:
                    medical_wer_scores.append(
                        compute_medical_wer(ref, hyp, medical_terms, weight=medical_weight)
                    )
            except Exception:
                continue

        results = {}
        if wer_scores:
            results["wer"] = sum(wer_scores) / len(wer_scores)
        if medical_wer_scores:
            results["medical_wer"] = sum(medical_wer_scores) / len(medical_wer_scores)

        return results

    return compute_metrics
