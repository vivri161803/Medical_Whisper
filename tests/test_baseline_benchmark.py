"""
test_baseline_benchmark.py — Smoke test per lo script 05_baseline_benchmark.py.

Verifica che:
- L'inferenza mlx_whisper funzioni su un audio sintetico minimo senza crash.
- Il report JSON prodotto sia valido e contenga i campi attesi.
"""

import json
import os

import numpy as np
import pytest
import soundfile as sf

# Import tramite importlib (necessario per il prefisso numerico 05_)
import importlib.util

_benchmark_path = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "05_baseline_benchmark.py"
)
_spec = importlib.util.spec_from_file_location("baseline_benchmark", _benchmark_path)
_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_benchmark)

transcribe_audio = _benchmark.transcribe_audio
run_benchmark = _benchmark.run_benchmark


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def short_audio(tmp_path):
    """Crea un breve audio sintetico (sine wave 1s a 16kHz)."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_path = tmp_path / "test_sine.wav"
    sf.write(str(audio_path), audio, samplerate=sr)
    return str(audio_path)


@pytest.fixture
def test_manifest(tmp_path, short_audio):
    """Crea un manifest di test con un singolo file audio."""
    manifest = [
        {
            "id": "test_sine_001",
            "audio_filepath": short_audio,
            "text": "questo è un test di trascrizione",
        }
    ]
    manifest_path = tmp_path / "test_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return str(manifest_path)


@pytest.fixture
def test_medical_terms(tmp_path):
    """Crea un glossario medico di test."""
    terms_path = tmp_path / "medical_terms.txt"
    terms_path.write_text("mandibola\nmascellare\nosteonecrosi\n")
    return str(terms_path)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestTranscribeAudio:
    """Test per la funzione di trascrizione."""

    @pytest.mark.slow
    def test_transcribe_returns_string(self, short_audio):
        """La trascrizione deve restituire una stringa (anche vuota per un sine wave)."""
        result = transcribe_audio(short_audio)
        assert isinstance(result, str)


class TestRunBenchmark:
    """Test per il benchmark completo."""

    @pytest.mark.slow
    def test_benchmark_produces_valid_report(
        self, test_manifest, test_medical_terms, tmp_path
    ):
        """
        Il benchmark deve produrre un report JSON valido con i campi attesi.
        """
        output_dir = str(tmp_path / "outputs")

        report = run_benchmark(
            manifest_path=test_manifest,
            medical_terms_path=test_medical_terms,
            output_dir=output_dir,
            use_wandb=False,
        )

        # Verifica struttura del report
        assert isinstance(report, dict)
        assert "timestamp" in report
        assert "model" in report
        assert "num_samples" in report
        assert "aggregate" in report
        assert "file_results" in report

        # Verifica che il file JSON sia stato salvato
        report_path = os.path.join(output_dir, "baseline_report.json")
        assert os.path.exists(report_path)

        # Verifica che il JSON sia valido
        with open(report_path, "r") as f:
            saved_report = json.load(f)
        assert isinstance(saved_report, dict)

        # Se ci sono risultati, verifica i campi
        if saved_report["file_results"]:
            result = saved_report["file_results"][0]
            assert "id" in result
            assert "wer" in result
            assert "medical_wer" in result
            assert isinstance(result["wer"], (int, float))
            assert isinstance(result["medical_wer"], (int, float))
