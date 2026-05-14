# Implementation Plan: Fase 4 — Fine-Tuning e Valutazione

Questo documento definisce il piano di implementazione iterativo per l'intera pipeline di fine-tuning MLX-native e valutazione Medical WER.

---

## Gruppo 1: Contratti Dati e Metriche (Fondamenta)

### 1.1 — Script `03_data_contracts.py`
- [ ] Definire i modelli Pydantic (`AudioSample`, `ManifestEntry`, `ValidationReport`) in `scripts/03_data_contracts.py`.
- [ ] Implementare validazione: sample rate = 16kHz, mono, durata ≤ 30s, testo non vuoto e privo di tag.
- [ ] Implementare funzione `validate_manifest(manifest_path) → ValidationReport` che itera sulle entry e produce un report strutturato (pass/fail + dettaglio errori).
- [ ] Esporre come CLI tramite `if __name__ == "__main__"` con argparse (path manifest in input).

### 1.2 — Script `04_metrics.py` (Medical WER)
- [ ] Implementare `compute_wer(reference, hypothesis) → float` usando `jiwer.wer()`.
- [ ] Implementare `load_medical_terms(path) → set[str]` per caricare il glossario da file esterno (un termine per riga, case-insensitive).
- [ ] Implementare `compute_medical_wer(reference, hypothesis, medical_terms, weight=3.0) → float`:
  - Usa `jiwer.process_words()` per ottenere l'alignment word-level.
  - Per ogni errore (sostituzione, cancellazione, inserzione), verifica se la parola coinvolta è nel glossario.
  - Se sì, moltiplica il conteggio di quell'errore per `weight`.
  - Ricalcola il WER pesato come `errori_pesati / totale_parole_reference`.
- [ ] Esporre entrambe le funzioni come API importabile per gli altri script.

### 1.3 — Test Gruppo 1 (Smoke Test)
- [ ] `tests/test_data_contracts.py`: generare rumore bianco 16kHz/mono/5s con `numpy`, salvare come `.wav` → deve passare la validazione. Testare anche un caso negativo (stereo, 48kHz).
- [ ] `tests/test_metrics.py`: creare reference/hypothesis fasulli con termini medici noti → verificare che `compute_wer` e `compute_medical_wer` producano float validi e che il Medical WER sia ≥ WER quando ci sono errori su termini medici.

---

## Gruppo 2: Baseline Benchmark (Il Punto Zero)

### 2.1 — Script `05_baseline_benchmark.py`
- [ ] Caricare il modello `whisper-small` via `mlx_whisper.transcribe()` per inferenza zero-shot.
- [ ] Iterare sui file audio del validation set (letti dal manifest).
- [ ] Per ogni file: trascrivere → calcolare WER e Medical WER → salvare risultato.
- [ ] Aggregare i risultati (media, mediana, min, max) e salvarli in `outputs/baseline_report.json`.
- [ ] Integrare logging su W&B (`wandb.init()` con tag `baseline`).
- [ ] CLI: argparse con `--manifest`, `--medical-terms`, `--output-dir`.

### 2.2 — Test Gruppo 2 (Smoke Test)
- [ ] `tests/test_baseline_benchmark.py`: generare un breve audio sintetico (sine wave 1s) e un manifest fittizio → verificare che lo script esegua l'inferenza senza crash e produca un report JSON valido. Non verificare l'accuratezza della trascrizione.

---

## Gruppo 3: Preprocessing Apple-Native

### 3.1 — Script `06_preprocess_mlx.py`
- [ ] Implementare `extract_log_mel(audio_path, n_mels=80, sr=16000) → mx.array` tramite `librosa.feature.melspectrogram` + conversione in `mlx.core.array`.
- [ ] Implementare `tokenize_text(text, tokenizer) → list[int]` usando il tokenizer di Whisper.
- [ ] Implementare `pad_or_trim_features(features, max_frames=3000) → mx.array` con padding a zero o trimming.
- [ ] Implementare `pad_labels(labels, max_length, pad_token=-100) → mx.array`.
- [ ] Implementare lo split train/val/test (80/10/10) con `seed=42` deterministico.
- [ ] Salvare le feature preprocessate in `data/preprocessed/{train,val,test}/` come file `.npz` (un file per sample o batch).
- [ ] CLI: argparse con `--manifest`, `--output-dir`, `--seed`, `--split-ratios`.

### 3.2 — Test Gruppo 3 (Smoke Test)
- [ ] `tests/test_preprocess_mlx.py`: generare audio sintetico (rumore bianco 3s) + testo fittizio → verificare che le feature log-Mel abbiano shape `(80, N)`, che il padding produca esattamente 3000 frame, e che i file `.npz` vengano creati correttamente. Verificare che lo split produca 3 sottocartelle non vuote.

---

## Gruppo 4: Configurazione e Training LoRA

### 4.1 — File `training_config.yaml`
- [ ] Creare il file YAML nella root del progetto con tutti gli iperparametri definiti in `requirements.md`.
- [ ] Implementare un loader `load_config(path) → dict` in un modulo condiviso (es. `scripts/config_loader.py`).

### 4.2 — Script `07_finetune_mlx.py` (Core)
- [ ] Caricare i pesi di `whisper-small` in formato MLX (via `mlx_whisper` o download diretto).
- [ ] Implementare `LoRALinear(nn.Module)`:
  - Parametri: `in_features`, `out_features`, `rank`, `alpha`, `dropout`.
  - Forward: `y = W_frozen @ x + (alpha/rank) * B @ A @ x`.
- [ ] Implementare `apply_lora(model, target_modules, rank, alpha, dropout)`:
  - Sostituire i layer `q_proj` e `v_proj` con `LoRALinear`.
  - Congelare il modello base con `model.freeze()`.
  - Scongelare solo i parametri LoRA.
- [ ] Implementare la loss function (cross-entropy sui token, ignorando `-100`).
- [ ] Costruire il training loop:
  ```
  loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
  optimizer = optim.AdamW(learning_rate=lr, weight_decay=wd)

  for epoch in range(num_epochs):
      for batch in data_loader:
          loss, grads = loss_and_grad_fn(model, batch)
          optimizer.update(model, grads)
          mx.eval(model.parameters(), optimizer.state)
          wandb.log({"train/loss": loss.item(), "train/step": step})
  ```
- [ ] Implementare evaluation loop intra-training:
  - Ogni `eval_every_n_steps`: fusione temporanea dei pesi LoRA → trascrizione val set → calcolo Medical WER.
  - Log su W&B: `val/wer`, `val/medical_wer`.
- [ ] Implementare checkpointing:
  - Ogni `save_every_n_steps`: salvataggio `adapters.npz` con `mx.savez()` contenente solo i parametri trainabili (`model.trainable_parameters()`).
  - Salvataggio finale di `adapters.npz` + `training_config.yaml` nella cartella `outputs/`.
- [ ] Integrare W&B: `wandb.init()` all'inizio, `wandb.log()` ad ogni step, `wandb.finish()` alla fine.
- [ ] CLI: argparse con `--config` (path al YAML), `--resume` (path opzionale a un checkpoint precedente).

### 4.3 — Test Gruppo 4 (Smoke Test)
- [ ] `tests/test_finetune_mlx.py`:
  - Creare un modello mock minimale con 1 layer di attention.
  - Applicare LoRA → verificare che solo i parametri LoRA siano trainabili.
  - Eseguire 2 iterazioni di training con dati sintetici randomici → verificare che la loss decresca o almeno non sia NaN.
  - Verificare che `adapters.npz` venga salvato e sia caricabile.

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
