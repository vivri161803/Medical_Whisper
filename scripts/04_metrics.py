"""
04_metrics.py — Wrapper CLI per le metriche di valutazione.

Le funzioni core sono in scripts/metrics.py.
Questo file mantiene la convenzione di naming numerato della pipeline.
"""

# Re-export da metrics.py per retrocompatibilità
from scripts.metrics import (
    compute_medical_wer,
    compute_wer,
    load_medical_terms,
)

__all__ = ["compute_wer", "compute_medical_wer", "load_medical_terms"]
