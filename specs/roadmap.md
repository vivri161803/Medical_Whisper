# Roadmap

La roadmap è organizzata in fasi di lavoro logiche e sequenziali, rispecchiando l'attuale struttura della pipeline del progetto. Essendo un Proof of Concept, il focus primario è sulla stabilità della preparazione del dato e sulla riproducibilità del training.

## Fase 1: Preparazione Testo (Completato)
- [x] Sviluppo dello script `01_prepare_text.py`.
- [x] Pulizia testuale dei file `.txt` estratti da PDF: rimozione sillabazioni a capo, spazi multipli, artefatti di formattazione.
- [x] Sentence tokenization con NLTK (lingua italiana).
- [x] Implementazione dell'algoritmo di Hybrid Chunking con vincoli: `min_words=15`, `max_words=35`, `max_chars=200` (limite XTTS_v2). Split delle frasi lunghe su virgole e punti e virgola.
- [x] Generazione di `data/synthetic_chunks/manifest_text.json` con i chunk testuali.

## Fase 2: Generazione Audio Sintetico (Completato)
- [x] Sviluppo dello script `02_generate_audio.py`.
- [x] Configurazione del modello Coqui XTTS_v2 locale con voice cloning da file di riferimento (`data/raw/reference_voice.wav`).
- [x] Generazione dei file `.wav` per ogni chunk testuale con meccanismo resume-safe.
- [x] Salvataggio del dataset audio sintetico in `data/synthetic_audio/` con relativo `manifest_synthetic.json`.
- [x] Validazione manuale a campione della pronuncia e della qualità audio.

## Fase 3 (Pivot): Data Augmentation (Completato)
- [x] Definizione dei modelli dati (Pydantic) per la validazione di manifest in ingresso e in uscita.
- [x] Implementazione della pipeline `ClassroomAugmenter` usando `audiomentations` (Reverb, Background Noise, BandPass, Volume Fluctuation).
- [x] Scrittura dei test (pytest) per verificare i requisiti spettrali, dinamici e i vincoli rigidi sulla durata dell'audio (<= 30s).
- [x] Integrazione dello script CLI tramite `Typer` per l'orchestrazione del processing batch.

## Fase 4: Fine-Tuning e Valutazione (Completato)

Pipeline MLX-native per il fine-tuning LoRA del modello `whisper-small` su Apple Silicon, con evaluation Medical WER e tracking W&B.

### A: Contratti Dati & Metriche (Completato)
- [x] **`03_data_contracts.py`:** Validazione Pydantic del dataset (16kHz, mono, ≤30s, testo pulito). AliasChoices per compatibilità manifest sintetico/augmented.
- [x] **`04_metrics.py` (Medical WER):** WER standard via `jiwer` + WER pesato con penalità 3x su glossario medico esterno.
- [x] **Test:** `test_data_contracts.py`, `test_metrics.py` — tutti passati.

### B: Baseline Benchmark (Completato)
- [x] **`05_baseline_benchmark.py`:** Inferenza zero-shot con `mlx_whisper.transcribe()`, calcolo WER/Medical WER, report JSON + logging W&B.
- [x] **Risultato baseline:** WER = 0.2279, Medical WER = 0.2301 (benchmark da abbattere).
- [x] **Test:** `test_baseline_benchmark.py` — passato.

### C: Preprocessing Apple-Native (Completato)
- [x] **`06_preprocess_mlx.py`:** Estrazione mel con `mlx_whisper.audio.log_mel_spectrogram()` (pipeline nativa, non librosa). Formato `(n_frames, n_mels)` = `(3000, 80)`. Tokenizzazione con `transformers.WhisperTokenizer`. Split 80/10/10 deterministico.
- [x] **Test:** `test_preprocess_mlx.py` — passato.

### D: Training LoRA (Completato)
- [x] **`training_config.yaml`:** Configurazione centralizzata degli iperparametri. `entity: null` per auto-detection W&B.
- [x] **`07_finetune_mlx.py`:** LoRA su layer `query`/`value` (72 layer, 3.5M params trainabili su 244M totali). Training loop con `nn.value_and_grad()`, cross-entropy con masking, gradient clipping, checkpointing `.safetensors`.
- [x] **Evaluation intra-training:** val_loss ogni 50 step, WER/Medical WER a fine epoca via trascrizione autoregressiva (`mlx_whisper.decoding.decode(fp16=False)`).
- [x] **W&B logging:** `train/loss`, `train/avg_loss`, `train/grad_norm`, `eval/loss`, `eval/wer`, `eval/medical_wer`, `best_medical_wer`.
- [x] **Test:** `test_finetune_mlx.py` — 17 test (10 strutturali + 7 WER evaluation) — tutti passati.

### E: Pulizia e Documentazione (Completato)
- [x] Aggiornamento `pyproject.toml`: rimosse dipendenze legacy (`accelerate`, `tensorboard`, `torchvision`, `torchcodec`).
- [x] Aggiornamento `README.md`: istruzioni W&B, comandi Step 1–7, parametri.
- [x] Aggiornamento spec: allineamento `plan.md`, `requirements.md`, `validation.md` all'implementazione reale.

### F: fine-tuning massivo
- [x] Implementare early stopping nel training dello script [text](../scripts/07_finetune_mlx.py)
- [x] Svuotare [text](../data/synthetic_audio) e fare ripartire lo script 02 per la sintesi vocale
- [x] Procedere con la pipeline descritta nel @README.md fino al fine-tuning del modello

### G: PyTorch Cloud Pipeline (Completato)
- [x] Porting del training loop da MLX a PyTorch + HuggingFace Seq2SeqTrainer + PEFT LoRA
- [x] Custom Dataset per caricamento file `.npz` preprocessati
- [x] Metriche standalone (WER + Medical WER) compatibili con Trainer
- [x] Config YAML dedicata per GPU cloud (fp16, gradient checkpointing, gradient accumulation)
- [x] `requirements.txt` con PyTorch 2.4.0 + CUDA
- [x] Script di deploy `deploy_runpod.sh` per RunPod L4
- [x] Supporto multi-modello: whisper-small / medium / large-v3
- [x] Early stopping + CLI overrides
- [x] Spec: `specs/2026-05-17-pytorch-cloud-finetuning/`
 
## Fase 5: Sperimentazione e Deploy (In Pianificazione)
- [ ] Fine-tuning di `whisper-medium` su GPU cloud RunPod L4 tramite `scripts_cloud/`.
- [ ] Valutazione formale del modello su un test set "hold-out" (lezioni universitarie inedite).
- [ ] Redazione di un report conclusivo del PoC, misurando quantitativamente il delta di accuratezza tra il modello Whisper zero-shot e il modello fine-tunato.
- [ ] (Opzionale) Sperimentazione con `whisper-large-v3` se le risorse lo consentono.
- [ ] (Opzionale) Implementazione di uno script CLI dedicato all'inferenza o di una UI semplificata (Gradio/Streamlit) per consentire agli studenti di medicina di trascrivere agilmente le proprie registrazioni.