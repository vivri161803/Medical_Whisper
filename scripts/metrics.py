"""
metrics.py — Metriche di valutazione: WER standard e Medical WER.

Modulo importabile con le funzioni core per il calcolo delle metriche.
Lo script numerato 04_metrics.py è un alias di questo modulo.

Fornisce:
- compute_wer(): WER standard via jiwer.
- load_medical_terms(): caricamento glossario medico da file esterno.
- compute_medical_wer(): WER pesato che penalizza gli errori su terminologia medica.
"""

from pathlib import Path

import jiwer


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
                # Supporto per termini multi-parola (es. "processo coronoideo")
                terms.add(line.lower())
    return terms


def _is_medical_token(token: str, medical_terms: set[str]) -> bool:
    """
    Verifica se un token è un termine medico, controllando sia il token
    singolo sia se fa parte di un termine multi-parola nel glossario.
    """
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

    Algoritmo:
    1. Usa jiwer.process_words() per ottenere l'alignment word-level.
    2. Per ogni allineamento, classifica gli errori.
    3. Se la parola coinvolta nell'errore è medica, il suo conteggio vale `weight`.
    4. Il Medical WER = errori_pesati / totale_parole_reference_pesate.

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

    # Contatori pesati
    weighted_errors = 0.0
    weighted_total = 0.0

    # Itera sull'alignment word-level
    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            # Le parole corrette contano normalmente nel totale
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
