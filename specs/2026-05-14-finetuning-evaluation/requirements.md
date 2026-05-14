# Requirements: Fase 4 — Fine-Tuning e Valutazione (MLX-Native)

## Scopo e Contesto

Implementare l'intera pipeline di fine-tuning del modello `whisper-small` in modo **nativo per Apple Silicon**, eliminando le dipendenze di PyTorch dal training loop e sostituendole con l'ecosistema MLX. L'obiettivo finale è abbattere il Word Error Rate (WER) sulla terminologia di chirurgia maxillo-facciale rispetto al modello zero-shot, misurando il progresso tramite una metrica custom **Medical WER** e tracciando l'addestramento in tempo reale via **Weights & Biases (wandb)**.

## Decisioni Architetturali

### D1: Stack MLX-Native
- Il training loop è costruito interamente con `mlx.core`, `mlx.nn`, `mlx.optimizers`.
- Nessuna dipendenza da `torch`, `torchaudio`, `accelerate`, `tensorboard` o `transformers` per gli script della Fase 4.
- Le dipendenze legacy (`accelerate`, `tensorboard`, `torchvision`, `torchcodec`) saranno deprecate nel `pyproject.toml` (mantenute solo se necessarie a script delle Fasi 1–3).

### D2: LoRA (Low-Rank Adaptation)
- Si applica LoRA esclusivamente ai layer `q_proj` e `v_proj` dell'encoder e decoder di Whisper.
- Configurazione default: `rank=32`, `alpha=64`, `dropout=0.05`.
- Solo i pesi LoRA sono addestrabili (`requires_grad = True`); il modello base è congelato tramite `model.freeze()`.

### D3: Modello Base
- **`whisper-small`** (multilingual) come checkpoint di partenza.
- I pesi sono caricati in formato MLX direttamente tramite `mlx-whisper` (conversione automatica da HuggingFace Hub).

### D4: Medical WER con Glossario Esterno
- Il glossario dei termini medici è un file di configurazione esterno (es. `data/medical_terms.txt`), il cui path è passato come argomento CLI.
- Gli errori sui termini medici hanno un **peso penalizzante maggiore** (default `3.0×`) rispetto a parole funzionali (articoli, congiunzioni, preposizioni).

### D5: Split dei Dati
- Lo split train/val/test è gestito dallo script `06_preprocess_mlx.py`.
- Proporzioni default: **80% train / 10% val / 10% test**.
- Lo split è deterministico (seed fisso) per garantire riproducibilità.

### D6: Logging via W&B
- Tutta la telemetria di training (loss, WER, Medical WER, learning rate) è tracciata su **Weights & Biases**.
- Fallback a logging CLI su stdout se `wandb` non è configurato (`WANDB_MODE=disabled`). 

## Specifiche Tecniche per Script

### `03_data_contracts.py` — Contratti Dati Pydantic

| Vincolo | Valore |
|---|---|
| Sample Rate | 16000 Hz (esatto) |
| Canali | Mono (1 canale) |
| Durata | ≤ 30.0 secondi |
| Testo | Non vuoto, ripulito da tag HTML/XML |
| Formato Audio | `.wav` |

- **Input:** Path ad una directory contenente audio `.wav` e un file `manifest.json`.
- **Output:** Report di validazione (pass/fail per ogni entry del manifest).
- **Modelli Pydantic:** `AudioSample`, `ManifestEntry`, `ValidationReport`.

### `04_metrics.py` — Medical WER

- **Dipendenze:** `jiwer` (`process_words`, alignment word-level).
- **Funzioni esposte:**
  - `compute_wer(reference, hypothesis) → float` — WER standard via `jiwer.wer()`.
  - `compute_medical_wer(reference, hypothesis, medical_terms, weight=3.0) → float` — WER pesato dove gli errori (sostituzioni, cancellazioni, inserzioni) su token presenti nel glossario medico sono moltiplicati per `weight`.
- **Config esterno:** path al file `medical_terms.txt` (un termine per riga).

### `05_baseline_benchmark.py` — Benchmark Zero-Shot

- **Dipendenze:** `mlx-whisper` per inferenza, `04_metrics.py` per la valutazione.
- **Input:** Subset di validazione audio (directory), manifest con ground truth, path glossario medico.
- **Output:** Report JSON con WER e Medical WER per ogni file + aggregati.
- **Modello:** `mlx-community/whisper-small-mlx` (o equivalente dal Hub).

### `06_preprocess_mlx.py` — Preprocessing e Split

- **Feature Extraction:** Log-Mel spettrogramma a 80 canali (configurazione standard Whisper), calcolato con `librosa` e salvato come `mlx.core.array` in file `.npz`.
- **Tokenizzazione:** Tramite il tokenizer di `whisper` (vocabolario multilingue).
- **Split:** 80/10/10 (train/val/test) con seed deterministico, salvato in sottocartelle `data/preprocessed/{train,val,test}/`.
- **Padding:** Frame audio allineati a 3000 (30 secondi × 100 frame/s). Token di padding etichettati come `-100` per essere ignorati dalla loss.

### `07_finetune_mlx.py` — Training Loop LoRA

- **Config:** Caricata da `training_config.yaml` (centralizzata).
- **Iperparametri default (M1 Pro 16GB):**

| Parametro | Valore |
|---|---|
| `batch_size` | 4 |
| `learning_rate` | 1e-5 |
| `num_epochs` | 3 |
| `lora_rank` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| `optimizer` | AdamW |
| `precision` | float16 |
| `eval_every_n_steps` | 50 |
| `save_every_n_steps` | 100 |
| `max_grad_norm` | 1.0 |

- **Training Loop:** `mlx.nn.value_and_grad()` → `optimizer.update()` → `mx.eval()`.
- **Evaluation intra-training:** Ogni `eval_every_n_steps`, fusione temporanea LoRA → inferenza sul val set → calcolo Medical WER.
- **Checkpointing:** Salvataggio di `adapters.npz` (solo pesi LoRA) ogni `save_every_n_steps`.
- **Logging:** Integrazione `wandb.log()` per loss, WER, Medical WER, learning rate ad ogni step.

### `training_config.yaml` — Configurazione Centralizzata

```yaml
model:
  name: "mlx-community/whisper-small-mlx"
  language: "it"

lora:
  rank: 32
  alpha: 64
  dropout: 0.05
  target_modules: ["q_proj", "v_proj"]

training:
  batch_size: 4
  learning_rate: 1.0e-5
  num_epochs: 3
  optimizer: "adamw"
  weight_decay: 0.01
  max_grad_norm: 1.0
  precision: "float16"

evaluation:
  eval_every_n_steps: 50
  save_every_n_steps: 100
  medical_terms_path: "data/medical_terms.txt"
  medical_weight: 3.0

data:
  preprocessed_dir: "data/preprocessed"
  output_dir: "outputs"
  seed: 42
  split_ratios: [0.8, 0.1, 0.1]

wandb:
  project: "whisper-medical-finetuning"
  entity: null  # your wandb username
  run_name: null  # auto-generated if null
```

## Dipendenze — Aggiornamento Stack

### Da aggiungere
| Pacchetto | Versione | Scopo |
|---|---|---|
| `wandb` | ≥0.27.0 | Tracking training/eval (già installato) |

### Da deprecare (rimuovere da `pyproject.toml`)
| Pacchetto | Motivazione |
|---|---|
| `accelerate` | Sostituito da MLX native training |
| `tensorboard` | Sostituito da W&B |
| `torchvision` | Non utilizzato nella pipeline |
| `torchcodec` | Non utilizzato nella pipeline |

### Da mantenere (necessari per Fasi 1–3)
| Pacchetto | Utilizzato da |
|---|---|
| `torch` | Silero VAD, WhisperX |
| `torchaudio` | Fasi 1–2 (resampling) |
| `transformers` | WhisperX (Fase 1) |
