import pytest
import numpy as np
import librosa
from pathlib import Path
from pydantic import ValidationError

from data.augmentation.models import AudioTextPair
from data.augmentation.pipeline import ClassroomAugmenter

def test_pydantic_duration_validation(tmp_path):
    # Setup un file fittizio per soddisfare FilePath
    dummy_audio = tmp_path / "dummy.wav"
    dummy_audio.touch()

    # Test durata valida
    valid_pair = AudioTextPair(
        id="test_01",
        audio_path=dummy_audio,
        transcript="Testo di prova",
        duration_sec=29.9
    )
    assert valid_pair.duration_sec == 29.9

    # Test durata invalida (> 30.0s)
    with pytest.raises(ValidationError):
        AudioTextPair(
            id="test_02",
            audio_path=dummy_audio,
            transcript="Testo invalido",
            duration_sec=30.1
        )

def test_pydantic_filepath_validation():
    # Test file non esistente
    with pytest.raises(ValidationError):
        AudioTextPair(
            id="test_03",
            audio_path=Path("non_existent.wav"),
            transcript="Testo",
            duration_sec=10.0
        )

def test_bandpass_spectral_validation():
    augmenter = ClassroomAugmenter(sample_rate=16000)
    # Genera white noise
    np.random.seed(42)
    white_noise = np.random.normal(0, 1, 16000 * 5).astype(np.float32)
    
    augmented_audio = augmenter.apply_bandpass(white_noise)
    
    # Calcola lo spettro tramite STFT e librosa
    stft = np.abs(librosa.stft(augmented_audio))
    freqs = librosa.fft_frequencies(sr=16000)
    
    # Trova indici < 300Hz e > 6000Hz
    idx_low = np.where(freqs < 300)[0]
    idx_high = np.where(freqs > 6000)[0]
    
    stft_orig = np.abs(librosa.stft(white_noise))
    
    energy_low_aug = np.mean(stft[idx_low, :])
    energy_low_orig = np.mean(stft_orig[idx_low, :])
    
    energy_high_aug = np.mean(stft[idx_high, :])
    energy_high_orig = np.mean(stft_orig[idx_high, :])
    
    # L'energia residua deve essere inferiore (filtro attenuante)
    assert energy_low_aug < energy_low_orig * 0.8
    assert energy_high_aug < energy_high_orig * 0.8

def test_dynamic_gain_fluctuation():
    augmenter = ClassroomAugmenter(sample_rate=16000)
    # Genera onda sinusoidale a 440Hz, durata sufficiente per una transition
    t = np.linspace(0, 5, 16000 * 5, endpoint=False)
    sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    augmented_audio = augmenter.apply_volume_fluctuation(sine_wave)
    
    # Calcola RMS a finestre
    rms_orig = librosa.feature.rms(y=sine_wave)[0]
    rms_aug = librosa.feature.rms(y=augmented_audio)[0]
    
    # L'originale ha RMS costante (a parte i bordi)
    # L'aumentato ha una variazione diversa a causa del gain transition
    assert abs(np.std(rms_aug) - np.std(rms_orig)) > 0.001

def test_duration_constraint_trim_pad():
    # Audio di 29.9 secondi
    augmenter = ClassroomAugmenter(sample_rate=16000)
    audio_len = int(16000 * 29.9)
    dummy_audio = np.zeros(audio_len, dtype=np.float32)
    
    # Simula padding o espansione passando a process() che forza il max a 30s
    augmented_audio = augmenter.process(dummy_audio)
    
    # La lunghezza non deve MAI superare i 30.0 secondi
    max_samples = 16000 * 30
    assert len(augmented_audio) <= max_samples
