"""
test_preprocess_mlx.py — Smoke test per 06_preprocess_mlx.py.
"""
import json, os
import numpy as np
import pytest
import soundfile as sf
import importlib.util

_p = os.path.join(os.path.dirname(__file__), "..", "scripts", "06_preprocess_mlx.py")
_s = importlib.util.spec_from_file_location("preprocess_mlx", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

extract_log_mel = _m.extract_log_mel
pad_or_trim_features = _m.pad_or_trim_features
pad_labels = _m.pad_labels
split_data = _m.split_data

@pytest.fixture
def synth_audio(tmp_path):
    audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    p = tmp_path / "noise.wav"
    sf.write(str(p), audio, samplerate=16000)
    return str(p)

@pytest.fixture
def dataset_10(tmp_path):
    entries = []
    for i in range(10):
        audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
        p = tmp_path / f"s_{i:03d}.wav"
        sf.write(str(p), audio, samplerate=16000)
        entries.append({"id": f"s_{i:03d}", "audio_filepath": str(p), "text": f"Test {i}"})
    return entries

class TestLogMel:
    def test_shape(self, synth_audio):
        f = extract_log_mel(synth_audio)
        assert f.shape[1] == 80 and f.shape[0] > 0  # (n_frames, n_mels)

    def test_no_nan(self, synth_audio):
        f = extract_log_mel(synth_audio)
        assert not np.any(np.isnan(f))

class TestPadding:
    def test_pad_to_3000(self):
        f = np.random.randn(100, 80).astype(np.float32)  # (n_frames, n_mels)
        assert pad_or_trim_features(f, 3000).shape == (3000, 80)

    def test_trim_to_3000(self):
        f = np.random.randn(5000, 80).astype(np.float32)  # (n_frames, n_mels)
        assert pad_or_trim_features(f, 3000).shape == (3000, 80)

    def test_labels_pad_minus100(self):
        r = pad_labels([1, 2, 3], 10, -100)
        assert len(r) == 10 and r[3] == -100

class TestSplit:
    def test_proportions(self, dataset_10):
        tr, va, te = split_data(dataset_10, (0.8, 0.1, 0.1), 42)
        assert len(tr) + len(va) + len(te) == 10
        assert len(tr) == 8

    def test_deterministic(self, dataset_10):
        a = split_data(dataset_10, seed=42)
        b = split_data(dataset_10, seed=42)
        assert [e["id"] for e in a[0]] == [e["id"] for e in b[0]]

    def test_all_non_empty(self, dataset_10):
        tr, va, te = split_data(dataset_10, (0.8, 0.1, 0.1), 42)
        assert all(len(s) > 0 for s in [tr, va, te])
