# Implementation Plan: Fase 4 — Fine-Tuning e Valutazione

Questo documento definisce il piano di implementazione iterativo per l'intera pipeline di fine-tuning MLX-native e valutazione Medical WER.

---

## Gruppo 1: Contratti Dati e Metriche (Fondamenta)

### 1.1 — Script `03_data_contracts.py`
- [x] Definire i modelli Pydantic (`AudioSample`, `ManifestEntry`, `ValidationReport`) in `scripts/03_data_contracts.py`.
- [x] Implementare validazione: sample rate = 16kHz, mono, durata ≤ 30s, testo non vuoto e privo di tag.
- [x] Implementare funzione `validate_manifest(manifest_path) → ValidationReport` che itera sulle entry e produce un report strutturato (pass/fail + dettaglio errori).
- [x] Esporre come CLI tramite `if __name__ == "__main__"` con argparse (path manifest in input).
- [x] Aggiunta AliasChoices per gestire sia i campi del manifest sintetico (`text`, `audio_filepath`) che quelli del manifest augmented (`transcript`, `audio_path`).
- [x] Fix model_validator per appendere errori invece di sovrascriverli.
- [x] Priorità a `augmented_audio_path` (16kHz) rispetto agli originali (24kHz).

### 1.2 — Script `04_metrics.py` (Medical WER)
- [x] Implementare `compute_wer(reference, hypothesis) → float` usando `jiwer.wer()`.
- [x] Implementare `load_medical_terms(path) → set[str]` per caricare il glossario da file esterno (un termine per riga, case-insensitive).
- [x] Implementare `compute_medical_wer(reference, hypothesis, medical_terms, weight=3.0) → float`:
  - Usa `jiwer.process_words()` per ottenere l'alignment word-level.
  - Per ogni errore (sostituzione, cancellazione, inserzione), verifica se la parola coinvolta è nel glossario.
  - Se sì, moltiplica il conteggio di quell'errore per `weight`.
  - Ricalcola il WER pesato come `errori_pesati / totale_parole_reference`.
- [x] Esporre entrambe le funzioni come API importabile (`scripts/metrics.py`) per gli altri script.

### 1.3 — Test Gruppo 1 (Smoke Test)
- [x] `tests/test_data_contracts.py`: generare rumore bianco 16kHz/mono/5s con `numpy`, salvare come `.wav` → deve passare la validazione. Testare anche un caso negativo (stereo, 48kHz).
- [x] `tests/test_metrics.py`: creare reference/hypothesis fasulli con termini medici noti → verificare che `compute_wer` e `compute_medical_wer` producano float validi e che il Medical WER sia ≥ WER quando ci sono errori su termini medici.

---

## Gruppo 2: Baseline Benchmark (Il Punto Zero)

### 2.1 — Script `05_baseline_benchmark.py`
- [x] Caricare il modello `whisper-small` via `mlx_whisper.transcribe()` per inferenza zero-shot.
- [x] Iterare sui file audio del validation set (letti dal manifest).
- [x] Per ogni file: trascrivere → calcolare WER e Medical WER → salvare risultato.
- [x] Aggregare i risultati (media, mediana, min, max) e salvarli in `outputs/baseline_report.json`.
- [x] Integrare logging su W&B (`wandb.init()` con tag `baseline`).
- [x] CLI: argparse con `--manifest`, `--medical-terms`, `--output-dir`.

### 2.2 — Test Gruppo 2 (Smoke Test)
- [x] `tests/test_baseline_benchmark.py`: generare un breve audio sintetico (sine wave 1s) e un manifest fittizio → verificare che lo script esegua l'inferenza senza crash e produca un report JSON valido. Non verificare l'accuratezza della trascrizione.

---

## Gruppo 3: Preprocessing Apple-Native

### 3.1 — Script `06_preprocess_mlx.py`
- [x] Implementare `extract_log_mel(audio_path, n_mels=80, sr=16000) → np.ndarray` tramite `mlx_whisper.audio.log_mel_spectrogram()` (pipeline nativa, non librosa) + conversione in numpy array.
- [x] Implementare `tokenize_text(text, tokenizer) → list[int]` usando il tokenizer di Whisper (`transformers.WhisperTokenizer`).
- [x] Implementare `pad_or_trim_features(features, max_frames=3000) → np.ndarray` con padding a zero o trimming.
- [x] Implementare `pad_labels(labels, max_length, pad_token=-100) → np.ndarray`.
- [x] Implementare lo split train/val/test (80/10/10) con `seed=42` deterministico.
- [x] Salvare le feature preprocessate in `data/preprocessed/{train,val,test}/` come file `.npz` (un file per sample).
- [x] CLI: argparse con `--manifest`, `--output-dir`, `--seed`, `--split-ratios`.

> **Nota implementativa:** Le mel features sono ora in formato `(n_frames, n_mels)` = `(3000, 80)` — il formato nativo di mlx-whisper — non `(80, 3000)` come inizialmente specificato. Questo evita trasposizioni runtime e garantisce compatibilità diretta con `mlx_whisper.decoding.decode()`.

### 3.2 — Test Gruppo 3 (Smoke Test)
- [x] `tests/test_preprocess_mlx.py`: generare audio sintetico (rumore bianco 3s) + testo fittizio → verificare che le feature log-Mel abbiano shape `(N, 80)` (formato nativo), che il padding produca esattamente `(3000, 80)`, e che i file `.npz` vengano creati correttamente. Verificare che lo split produca 3 sottocartelle non vuote.

---

## Gruppo 4: Configurazione e Training LoRA

### 4.1 — File `training_config.yaml`
- [x] Creare il file YAML nella root del progetto con tutti gli iperparametri definiti in `requirements.md`.
- [x] Implementare un loader `load_config(path) → dict` in `scripts/config_loader.py`.
- [x] Fix: `entity: null` per auto-detection organizzazione W&B.

### 4.2 — Script `07_finetune_mlx.py` (Core)
- [x] Caricare i pesi di `whisper-small` in formato MLX via `mlx_whisper.load_models.load_model()`.
- [x] Implementare `LoRALinear(nn.Module)`:
  - Parametri: `in_features`, `out_features`, `rank`, `alpha`, `dropout`.
  - Forward: `y = W_frozen @ x + (alpha/rank) * B @ A @ x`.
- [x] Implementare `apply_lora(model, target_modules, rank, alpha, dropout)`:
  - Sostituire i layer `query` e `value` con `LoRALinear`.
  - Congelare il modello base con `model.freeze()`.
  - Scongelare solo i parametri LoRA.
- [x] Implementare la loss function (cross-entropy sui token, ignorando `-100`).
- [x] Costruire il training loop con `nn.value_and_grad()` → `optimizer.update()` → `mx.eval()`.
- [x] Implementare evaluation loop intra-training:
  - Ogni `eval_every_n_steps`: calcolo val_loss.
  - **A fine epoca**: trascrizione autoregressiva val set tramite `mlx_whisper.decoding.decode()` → calcolo WER e Medical WER.
  - Log su W&B: `eval/loss`, `eval/wer`, `eval/medical_wer`.
- [x] Implementare checkpointing:
  - Ogni `save_every_n_steps`: salvataggio `adapters_stepN.safetensors` con `mx.save_safetensors()` contenente solo i parametri trainabili.
  - Salvataggio finale di `adapters_final.safetensors` + `training_config.yaml` nella cartella `outputs/`.
- [x] Integrare W&B: `wandb.init()` all'inizio, `wandb.log()` ad ogni step, `wandb.finish()` alla fine.
- [x] CLI: argparse con `--config` (path al YAML).

> **Nota implementativa critica — LoRA target modules:**
> Le spec originali indicavano `q_proj` e `v_proj`, ma il modello `whisper-small-mlx` usa i nomi `query` e `value` per i layer di attenzione. Questa discrepanza causava 0 layer LoRA applicati.

> **Nota implementativa critica — Mel features:**
> Le mel features devono essere calcolate con `mlx_whisper.audio.log_mel_spectrogram()`, non con librosa. Librosa produce mel in scala `[-80, 0]` dB, mentre mlx-whisper usa una scala normalizzata `[-1, 2]`. Usare la pipeline sbagliata produce WER = 1.0.

> **Nota implementativa critica — Decode fp16:**
> Il decoder di `mlx_whisper.decoding.decode()` verifica il dtype delle audio features. Con modello in float32, serve `fp16=False` nel `DecodingOptions`, altrimenti il decode fallisce silenziosamente.

### 4.3 — Test Gruppo 4 (Smoke Test + WER Evaluation)
- [x] `tests/test_finetune_mlx.py` — Test strutturali:
  - Creare un modello mock minimale con 1 layer di attention.
  - Applicare LoRA → verificare che solo i parametri LoRA siano trainabili.
  - Eseguire 2 iterazioni di training con dati sintetici randomici → verificare che la loss non sia NaN.
  - Verificare che `adapters.npz` venga salvato e sia caricabile.
- [x] `tests/test_finetune_mlx.py::TestWEREvaluation` — Test WER evaluation (7 test):
  - `test_mel_format_is_native`: shape `(3000, 80)` non `(80, 3000)`.
  - `test_mel_scale_is_native`: range `~[-1, 2]` non `[-80, 0]`.
  - `test_reference_decode_produces_text`: labels → testo non vuoto.
  - `test_whisper_decode_fp16_false`: decode funziona con `fp16=False`.
  - `test_whisper_decode_fp16_true_raises`: `fp16=True` → TypeError (regression guard).
  - `test_single_sample_wer_below_one`: WER < 1.0 su singolo campione reale.
  - `test_compute_epoch_wer_returns_metrics`: pipeline end-to-end WER + Medical WER < 1.0.

---

## Gruppo 5: Pulizia Dipendenze e Documentazione

### 5.1 — Aggiornamento `pyproject.toml`
- [ ] Rimuovere: `accelerate`, `tensorboard`, `torchvision`, `torchcodec`.
- [ ] Verificare che nessuno script delle Fasi 1–3 dipenda dai pacchetti rimossi.

### 5.2 — Aggiornamento `tech-stack.md`
- [ ] Aggiungere sezione MLX-native (mlx, mlx-whisper, wandb).
- [ ] Marcare come deprecati i pacchetti rimossi.

### 5.3 — Aggiornamento `README.md`
- [ ] Aggiungere sezione "Weights & Biases Setup" con istruzioni per:
  - Installazione (`uv add wandb`, già fatto).
  - Login (`wandb login`).
  - Configurazione nel `training_config.yaml`.
  - Modalità offline (`WANDB_MODE=disabled`).
- [ ] Aggiungere sezione "Fase 4: Fine-Tuning" con comandi per eseguire gli script 03–07 in sequenza.

---

## Ordine di Esecuzione

```
Gruppo 1 (Contratti + Metriche)
    ↓
Gruppo 2 (Baseline Benchmark)
    ↓
Gruppo 3 (Preprocessing)
    ↓
Gruppo 4 (Training LoRA)
    ↓
Gruppo 5 (Cleanup + Docs)
```

Ogni gruppo è auto-contenuto: i test devono passare prima di procedere al gruppo successivo.
