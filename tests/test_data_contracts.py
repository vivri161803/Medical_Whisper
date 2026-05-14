"""
test_data_contracts.py — Smoke test per lo script 03_data_contracts.py.

Verifica che:
- Un file audio valido (rumore bianco, 16kHz, mono, 5s) superi la validazione.
- Un file audio non valido (stereo, 48kHz) venga rifiutato con errori specifici.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

# Import tramite importlib dato il prefisso numerico
import importlib.util

_contracts_path = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "03_data_contracts.py"
)
_spec = importlib.util.spec_from_file_location("data_contracts", _contracts_path)
_contracts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contracts)

validate_manifest = _contracts.validate_manifest
validate_audio_file = _contracts.validate_audio_file
ManifestEntry = _contracts.ManifestEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_audio_dir(tmp_path):
    """Crea un file audio valido (16kHz, mono, 5s di rumore bianco)."""
    audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1  # 5 secondi
    audio_path = tmp_path / "valid_audio.wav"
    sf.write(str(audio_path), audio, samplerate=16000, subtype="PCM_16")
    return str(audio_path)


@pytest.fixture
def invalid_audio_dir(tmp_path):
    """Crea un file audio non valido (48kHz, stereo, 35s)."""
    audio = np.random.randn(48000 * 35, 2).astype(np.float32) * 0.1  # stereo, 35s
    audio_path = tmp_path / "invalid_audio.wav"
    sf.write(str(audio_path), audio, samplerate=48000, subtype="PCM_16")
    return str(audio_path)


@pytest.fixture
def valid_manifest(tmp_path, valid_audio_dir):
    """Crea un manifest JSON valido con un singolo file audio."""
    manifest = [
        {
            "id": "test_001",
            "audio_filepath": valid_audio_dir,
            "text": "La mandibola è un osso impari e mediano.",
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return str(manifest_path)


@pytest.fixture
def invalid_manifest(tmp_path, invalid_audio_dir):
    """Crea un manifest con un file audio non valido."""
    manifest = [
        {
            "id": "test_bad_001",
            "audio_filepath": invalid_audio_dir,
            "text": "Test con audio non valido.",
        }
    ]
    manifest_path = tmp_path / "manifest_invalid.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return str(manifest_path)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestValidAudio:
    """Test per campioni audio validi."""

    def test_valid_audio_passes_validation(self, valid_audio_dir):
        """Un file 16kHz mono ≤30s deve superare la validazione."""
        result = validate_audio_file("test_001", valid_audio_dir)
        assert result.is_valid is True
        assert result.errors == []
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.duration_sec <= 30.0

    def test_valid_manifest_all_pass(self, valid_manifest):
        """Un manifest con entry valide deve avere all_passed = True."""
        report = validate_manifest(valid_manifest)
        assert report.all_passed is True
        assert report.valid_entries == 1
        assert report.invalid_entries == 0


class TestInvalidAudio:
    """Test per campioni audio non validi."""

    def test_stereo_48khz_fails(self, invalid_audio_dir):
        """Un file stereo a 48kHz e >30s deve fallire con errori specifici."""
        result = validate_audio_file("test_bad_001", invalid_audio_dir)
        assert result.is_valid is False
        assert len(result.errors) >= 2  # almeno sample_rate + channels
        # Verifica che gli errori menzionino i problemi
        error_text = " ".join(result.errors)
        assert "48000" in error_text or "Sample rate" in error_text
        assert "2" in error_text or "mono" in error_text.lower()

    def test_invalid_manifest_fails(self, invalid_manifest):
        """Un manifest con entry non valide deve riportare errori."""
        report = validate_manifest(invalid_manifest)
        assert report.all_passed is False
        assert report.invalid_entries >= 1


class TestManifestEntry:
    """Test per la validazione delle entry del manifest."""

    def test_empty_text_raises(self):
        """Un testo vuoto deve sollevare un errore."""
        with pytest.raises(Exception):
            ManifestEntry(id="test", audio_filepath="/dummy.wav", text="")

    def test_html_tags_raise(self):
        """Un testo con tag HTML deve sollevare un errore."""
        with pytest.raises(Exception):
            ManifestEntry(
                id="test",
                audio_filepath="/dummy.wav",
                text="<p>Testo con tag</p>",
            )

    def test_clean_text_passes(self):
        """Un testo pulito deve essere accettato."""
        entry = ManifestEntry(
            id="test",
            audio_filepath="/dummy.wav",
            text="Il processo coronoideo della mandibola.",
        )
        assert entry.text == "Il processo coronoideo della mandibola."
