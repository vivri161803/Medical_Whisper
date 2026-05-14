"""
test_metrics.py — Smoke test per lo script 04_metrics.py.

Verifica che:
- compute_wer() produca float validi.
- compute_medical_wer() penalizzi maggiormente gli errori su termini medici.
- load_medical_terms() carichi correttamente un glossario.
"""

import os
import tempfile

import pytest

from scripts.metrics import compute_wer, compute_medical_wer, load_medical_terms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def medical_terms_file(tmp_path):
    """Crea un file di glossario medico temporaneo."""
    terms = [
        "mandibola",
        "mascellare",
        "osteonecrosi",
        "bifosfonati",
        "processo coronoideo",
        "articolazione temporomandibolare",
    ]
    terms_path = tmp_path / "medical_terms.txt"
    terms_path.write_text("\n".join(terms))
    return str(terms_path)


@pytest.fixture
def medical_terms_set():
    """Set di termini medici per i test."""
    return {
        "mandibola",
        "mascellare",
        "osteonecrosi",
        "bifosfonati",
        "processo coronoideo",
        "articolazione temporomandibolare",
    }


# ---------------------------------------------------------------------------
# Test WER Standard
# ---------------------------------------------------------------------------

class TestComputeWER:
    """Test per la funzione compute_wer."""

    def test_identical_strings_zero_wer(self):
        """Stringhe identiche devono avere WER = 0."""
        wer = compute_wer(
            "la mandibola è un osso impari",
            "la mandibola è un osso impari",
        )
        assert wer == 0.0

    def test_different_strings_positive_wer(self):
        """Stringhe diverse devono avere WER > 0."""
        wer = compute_wer(
            "la mandibola è un osso impari",
            "la mandibula è un osso pari",
        )
        assert isinstance(wer, float)
        assert wer > 0.0

    def test_empty_hypothesis_full_wer(self):
        """Ipotesi vuota deve avere WER = 1.0 (tutte le parole mancanti)."""
        wer = compute_wer(
            "la mandibola è un osso impari",
            "",
        )
        assert wer == 1.0

    def test_empty_reference_raises(self):
        """Reference vuoto deve sollevare un errore."""
        with pytest.raises(ValueError):
            compute_wer("", "qualcosa")


# ---------------------------------------------------------------------------
# Test Medical WER
# ---------------------------------------------------------------------------

class TestComputeMedicalWER:
    """Test per la funzione compute_medical_wer."""

    def test_medical_error_penalized_more(self, medical_terms_set):
        """
        Un errore su un termine medico deve produrre un Medical WER ≥ WER standard.
        """
        reference = "la mandibola è un osso impari e mediano"
        # Errore sul termine medico "mandibola" → "mandibula"
        hypothesis_medical = "la mandibula è un osso impari e mediano"

        wer_standard = compute_wer(reference, hypothesis_medical)
        medical_wer = compute_medical_wer(
            reference, hypothesis_medical, medical_terms_set, weight=3.0
        )

        assert isinstance(medical_wer, float)
        assert medical_wer >= wer_standard

    def test_non_medical_error_normal_weight(self, medical_terms_set):
        """
        Un errore su una congiunzione non dovrebbe avere penalizzazione extra.
        """
        reference = "la mandibola e il mascellare sono collegati"
        # Errore sulla congiunzione "e" → "o"
        hypothesis = "la mandibola o il mascellare sono collegati"

        wer_standard = compute_wer(reference, hypothesis)
        medical_wer = compute_medical_wer(
            reference, hypothesis, medical_terms_set, weight=3.0
        )

        # Il Medical WER dovrebbe essere simile al WER standard
        # dato che l'errore non è su un termine medico
        assert isinstance(medical_wer, float)
        assert medical_wer > 0.0

    def test_no_errors_zero_medical_wer(self, medical_terms_set):
        """Senza errori, il Medical WER deve essere 0."""
        reference = "la mandibola è un osso impari"
        medical_wer = compute_medical_wer(
            reference, reference, medical_terms_set, weight=3.0
        )
        assert medical_wer == 0.0

    def test_returns_float(self, medical_terms_set):
        """Il Medical WER deve restituire un float."""
        result = compute_medical_wer(
            "test con mandibola",
            "test senza mandibula",
            medical_terms_set,
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Test Glossario
# ---------------------------------------------------------------------------

class TestLoadMedicalTerms:
    """Test per il caricamento del glossario."""

    def test_loads_terms_correctly(self, medical_terms_file):
        """Il glossario deve essere caricato come set di termini lowercase."""
        terms = load_medical_terms(medical_terms_file)
        assert isinstance(terms, set)
        assert "mandibola" in terms
        assert "osteonecrosi" in terms
        assert len(terms) == 6

    def test_ignores_comments_and_empty_lines(self, tmp_path):
        """Righe vuote e commenti (#) devono essere ignorate."""
        terms_path = tmp_path / "terms_with_comments.txt"
        terms_path.write_text("# Commento\nmandibola\n\n# Altro commento\nmascellare\n")
        terms = load_medical_terms(str(terms_path))
        assert len(terms) == 2
        assert "mandibola" in terms

    def test_case_insensitive(self, tmp_path):
        """I termini devono essere salvati in minuscolo."""
        terms_path = tmp_path / "terms_case.txt"
        terms_path.write_text("Mandibola\nMASCELLARE\nosteonecrosi\n")
        terms = load_medical_terms(str(terms_path))
        assert "mandibola" in terms
        assert "mascellare" in terms

    def test_file_not_found_raises(self):
        """Un path inesistente deve sollevare FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_medical_terms("/path/inesistente/terms.txt")
