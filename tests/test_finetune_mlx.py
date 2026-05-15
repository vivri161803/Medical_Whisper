"""
test_finetune_mlx.py — Smoke test per 07_finetune_mlx.py.

Verifica che:
- LoRALinear funzioni correttamente.
- apply_lora congeli i pesi base e mantenga trainabili solo i parametri LoRA.
- 2 iterazioni di training con dati sintetici non producano NaN.
- adapters.npz venga salvato e ricaricato correttamente.
"""
import math
import os
import tempfile

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

import importlib.util

_ft_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "07_finetune_mlx.py")
_spec = importlib.util.spec_from_file_location("finetune_mlx", _ft_path)
_ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft)

LoRALinear = _ft.LoRALinear
count_parameters = _ft.count_parameters


# ---------------------------------------------------------------------------
# Modello Mock minimale con q_proj e v_proj
# ---------------------------------------------------------------------------

class MockAttention(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x):
        q = self.q_proj(x)
        v = self.v_proj(x)
        return self.out_proj(q + v)


class MockModel(nn.Module):
    def __init__(self, dim: int = 64, vocab_size: int = 100):
        super().__init__()
        self.attention = MockAttention(dim)
        self.head = nn.Linear(dim, vocab_size)

    def __call__(self, x):
        h = self.attention(x)
        return self.head(h)


# ---------------------------------------------------------------------------
# Test LoRALinear
# ---------------------------------------------------------------------------

class TestLoRALinear:
    def test_forward_shape(self):
        layer = LoRALinear(64, 64, rank=4, alpha=8.0)
        x = mx.random.normal((2, 64))
        y = layer(x)
        mx.eval(y)
        assert y.shape == (2, 64)

    def test_output_not_nan(self):
        layer = LoRALinear(64, 64, rank=4, alpha=8.0, dropout=0.0)
        x = mx.random.normal((2, 64))
        y = layer(x)
        mx.eval(y)
        assert not mx.any(mx.isnan(y)).item()


# ---------------------------------------------------------------------------
# Test LoRA Application
# ---------------------------------------------------------------------------

class TestApplyLoRA:
    def test_freeze_and_unfreeze(self):
        """Solo i parametri LoRA (lora_a, lora_b) devono essere trainabili."""
        model = MockModel(dim=64, vocab_size=100)
        mx.eval(model.parameters())

        # Congela tutto
        model.freeze()

        # Sostituisci q_proj e v_proj con LoRALinear
        for name in ["q_proj", "v_proj"]:
            original = getattr(model.attention, name)
            lora_layer = LoRALinear(
                in_features=original.weight.shape[1],
                out_features=original.weight.shape[0],
                rank=4, alpha=8.0, dropout=0.0, bias=True,
            )
            lora_layer.weight = original.weight
            if original.bias is not None:
                lora_layer.bias = original.bias
            lora_layer.freeze()
            lora_layer.unfreeze(keys=["lora_a", "lora_b"])
            setattr(model.attention, name, lora_layer)

        # Verifica parametri trainabili
        total, trainable = count_parameters(model)
        assert trainable > 0
        assert trainable < total
        ratio = trainable / total
        assert ratio < 0.15  # LoRA deve essere una piccola frazione


# ---------------------------------------------------------------------------
# Test Training Loop (2 iterazioni)
# ---------------------------------------------------------------------------

class TestTrainingLoop:
    def test_two_steps_no_nan(self):
        """2 step di training devono produrre una loss finita (non NaN)."""
        model = MockModel(dim=64, vocab_size=100)
        mx.eval(model.parameters())

        # Applica LoRA manualmente
        model.freeze()
        for name in ["q_proj", "v_proj"]:
            original = getattr(model.attention, name)
            lora = LoRALinear(64, 64, rank=4, alpha=8.0, dropout=0.0, bias=True)
            lora.weight = original.weight
            if original.bias is not None:
                lora.bias = original.bias
            lora.freeze()
            lora.unfreeze(keys=["lora_a", "lora_b"])
            setattr(model.attention, name, lora)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        optimizer = optim.AdamW(learning_rate=1e-3)
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        losses = []
        for _ in range(2):
            x = mx.random.normal((4, 64))
            y = mx.random.randint(0, 100, (4,))
            loss, grads = loss_and_grad_fn(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(loss.item())

        assert all(not math.isnan(l) for l in losses)
        assert all(not math.isinf(l) for l in losses)


# ---------------------------------------------------------------------------
# Test Checkpointing
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def test_save_and_load_adapters(self, tmp_path):
        """adapters.npz deve essere salvato e ricaricabile."""
        model = MockModel(dim=64, vocab_size=100)
        mx.eval(model.parameters())
        model.freeze()

        for name in ["q_proj", "v_proj"]:
            original = getattr(model.attention, name)
            lora = LoRALinear(64, 64, rank=4, alpha=8.0, dropout=0.0, bias=True)
            lora.weight = original.weight
            lora.freeze()
            lora.unfreeze(keys=["lora_a", "lora_b"])
            setattr(model.attention, name, lora)

        # Salva solo i parametri trainabili
        adapter_path = str(tmp_path / "adapters.npz")
        trainable = {}

        def _flatten(params, prefix=""):
            if isinstance(params, dict):
                for k, val in params.items():
                    _flatten(val, f"{prefix}.{k}" if prefix else k)
            elif isinstance(params, list):
                for i, val in enumerate(params):
                    _flatten(val, f"{prefix}.{i}")
            elif isinstance(params, mx.array):
                trainable[prefix] = params

        _flatten(model.trainable_parameters())
        mx.savez(adapter_path, **trainable)

        assert os.path.exists(adapter_path)

        # Ricarica
        loaded = dict(mx.load(adapter_path))
        assert len(loaded) > 0
        for key in loaded:
            assert "lora" in key


# ---------------------------------------------------------------------------
# Test WER Evaluation
# ---------------------------------------------------------------------------

compute_epoch_wer = _ft.compute_epoch_wer


class TestWEREvaluation:
    """Test per la funzione compute_epoch_wer — verifica che la pipeline
    di trascrizione autoregressiva + calcolo metriche funzioni end-to-end."""

    @pytest.fixture(scope="class")
    def whisper_model(self):
        """Carica il modello Whisper una sola volta per tutti i test della classe."""
        from mlx_whisper.load_models import load_model
        model = load_model("mlx-community/whisper-small-mlx", dtype=mx.float32)
        mx.eval(model.parameters())
        return model

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Carica il tokenizer Whisper."""
        from transformers import WhisperTokenizer
        return WhisperTokenizer.from_pretrained(
            "openai/whisper-small", language="it", task="transcribe"
        )

    @pytest.fixture(scope="class")
    def val_files(self):
        """Restituisce i file di validazione preprocessati."""
        val_dir = "data/preprocessed/val"
        if not os.path.exists(val_dir):
            pytest.skip("data/preprocessed/val non trovata — esegui 06_preprocess_mlx.py prima")
        files = sorted([
            os.path.join(val_dir, f)
            for f in os.listdir(val_dir)
            if f.endswith(".npz")
        ])
        if not files:
            pytest.skip("Nessun file .npz in data/preprocessed/val")
        return files

    @pytest.fixture(scope="class")
    def medical_terms(self):
        """Carica i termini medici."""
        from scripts.metrics import load_medical_terms
        path = "data/medical_terms.txt"
        if not os.path.exists(path):
            return set()
        return load_medical_terms(path)

    def test_mel_format_is_native(self, val_files):
        """Le mel features devono essere in formato (n_frames, n_mels) = (3000, 80)."""
        data = np.load(val_files[0])
        mel = data["log_mel"]
        assert mel.shape == (3000, 80), (
            f"Mel shape {mel.shape} non è nel formato nativo mlx-whisper (3000, 80). "
            f"Riesegui 06_preprocess_mlx.py."
        )

    def test_mel_scale_is_native(self, val_files):
        """Le mel features devono essere nella scala nativa mlx-whisper (range ~[-1, 2])."""
        data = np.load(val_files[0])
        mel = data["log_mel"]
        # La scala nativa è circa [-1, 2], non [-80, 0] (librosa) né [0, 1] (normalizzato)
        assert mel.min() > -5.0, f"Mel min={mel.min():.2f} — sembra scala librosa, non nativa"
        assert mel.max() < 5.0, f"Mel max={mel.max():.2f} — fuori dal range atteso"

    def test_reference_decode_produces_text(self, val_files, tokenizer):
        """I token di riferimento devono decodificare in testo non vuoto."""
        data = np.load(val_files[0])
        labels = data["labels"]
        ref_tokens = [int(t) for t in labels if t != -100 and t < 50257]
        reference = tokenizer.decode(ref_tokens, skip_special_tokens=True).strip()
        assert len(reference) > 0, "Reference decodificata vuota — problema nei labels preprocessati"
        assert len(reference.split()) >= 3, f"Reference troppo corta: '{reference}'"

    def test_whisper_decode_fp16_false(self, whisper_model, val_files):
        """Il decode con fp16=False deve funzionare con modello float32."""
        from mlx_whisper.decoding import DecodingOptions, decode as whisper_decode

        data = np.load(val_files[0])
        mel = mx.array(data["log_mel"])

        options = DecodingOptions(
            language="it", task="transcribe",
            without_timestamps=True, fp16=False,
        )
        # Non deve lanciare eccezioni
        result = whisper_decode(whisper_model, mel, options)
        hyp = result[0].text if isinstance(result, list) else result.text
        assert len(hyp.strip()) > 0, "Decode ha prodotto testo vuoto"

    def test_whisper_decode_fp16_true_raises(self, whisper_model, val_files):
        """Il decode con fp16=True deve fallire con modello float32 (encoder produce f32)."""
        from mlx_whisper.decoding import DecodingOptions, decode as whisper_decode

        data = np.load(val_files[0])
        mel = mx.array(data["log_mel"])

        options = DecodingOptions(
            language="it", task="transcribe",
            without_timestamps=True, fp16=True,
        )
        with pytest.raises(TypeError, match="incorrect dtype"):
            whisper_decode(whisper_model, mel, options)

    def test_single_sample_wer_below_one(self, whisper_model, val_files, tokenizer):
        """La WER su un singolo campione deve essere < 1.0 (il modello capisce il testo)."""
        from mlx_whisper.decoding import DecodingOptions, decode as whisper_decode
        from scripts.metrics import compute_wer

        data = np.load(val_files[0])
        mel = mx.array(data["log_mel"])
        labels = data["labels"]

        ref_tokens = [int(t) for t in labels if t != -100 and t < 50257]
        reference = tokenizer.decode(ref_tokens, skip_special_tokens=True).strip()

        options = DecodingOptions(
            language="it", task="transcribe",
            without_timestamps=True, fp16=False,
        )
        result = whisper_decode(whisper_model, mel, options)
        hyp = result[0].text if isinstance(result, list) else result.text

        wer = compute_wer(reference, hyp.strip())
        assert wer < 1.0, (
            f"WER={wer:.4f} — il modello non sta trascrivendo.\n"
            f"  Reference:  '{reference[:100]}'\n"
            f"  Hypothesis: '{hyp[:100]}'"
        )

    def test_compute_epoch_wer_returns_metrics(
        self, whisper_model, val_files, tokenizer, medical_terms
    ):
        """compute_epoch_wer deve restituire eval/wer e eval/medical_wer < 1.0."""
        # Testa su un sottoinsieme per velocità
        subset = val_files[:3]
        results = compute_epoch_wer(
            whisper_model, subset, tokenizer,
            medical_terms, medical_weight=3.0,
        )

        assert "eval/wer" in results, "compute_epoch_wer non ha restituito eval/wer"
        assert "eval/medical_wer" in results, "compute_epoch_wer non ha restituito eval/medical_wer"
        assert results["eval/wer"] < 1.0, f"WER media={results['eval/wer']:.4f} — ancora 1.0!"
        assert results["eval/medical_wer"] < 1.0, (
            f"Medical WER={results['eval/medical_wer']:.4f} — ancora 1.0!"
        )

