# 🏥 Whisper Medical — Fine-Tuning per Terminologia Medica Italiana

Pipeline per il fine-tuning di OpenAI Whisper, specializzata nel riconoscimento del gergo tecnico medico italiano. I modelli open source (come Whisper Medium) tendono a fallire su terminologia clinica, anatomica e farmacologica — questo progetto risolve il problema addestrando il modello su trascrizioni manuali verificate da studenti di medicina.

## 📋 Panoramica

La pipeline si compone di **7 step** sequenziali:

| Step | Script | Descrizione |
|------|--------|-------------|
| 1 | `scripts/01_prepare_text.py` | Pulizia testi raw e chunking per TTS |
| 2 | `scripts/02_generate_audio.py` | Generazione audio sintetico con XTTS_v2 |
| 2.5 | `data/augmentation/cli.py` | Data Augmentation acustica (rumore aula, riverbero) |
| 3 | `scripts/03_data_contracts.py` | Validazione Pydantic del dataset (16kHz, mono, ≤30s) |
| 4 | `scripts/04_metrics.py` | WER standard + Medical WER pesato |
| 5 | `scripts/05_baseline_benchmark.py` | Benchmark zero-shot con mlx-whisper |
| 6 | `scripts/06_preprocess_mlx.py` | Estrazione log-Mel + split train/val/test |
| 7 | `scripts/07_finetune_mlx.py` | Fine-tuning LoRA nativo MLX con W&B |

## 📁 Struttura del Progetto

```
Whisper/
├── scripts/
│   ├── 01_prepare_text.py        # Pulizia testi + chunking
│   ├── 02_generate_audio.py      # Generazione audio sintetico
│   ├── 03_data_contracts.py      # Validazione Pydantic dataset
│   ├── 04_metrics.py             # WER + Medical WER
│   ├── 05_baseline_benchmark.py  # Benchmark zero-shot
│   ├── 06_preprocess_mlx.py      # Feature extraction + split
│   ├── 07_finetune_mlx.py        # Fine-tuning LoRA MLX
│   └── config_loader.py          # Loader per training_config.yaml
├── training_config.yaml          # Iperparametri centralizzati
├── data/
│   ├── raw/                      # Audio + .txt originali
│   ├── synthetic_audio/          # Dataset sintetici (TTS)
│   ├── augmented_audio/          # Dataset con augmentation
│   ├── preprocessed/             # Feature log-Mel (train/val/test)
│   ├── medical_terms.txt         # Glossario termini medici
│   └── augmentation/             # Pipeline augmentation
├── outputs/                      # Adapter LoRA + report
├── tests/                        # Smoke test per ogni script
├── specs/                        # Spec-Driven Development docs
├── pyproject.toml
└── README.md
```

## 🔧 Requisiti di Sistema

- **Hardware**: Mac con Apple Silicon (M1 Pro testato) — usa backend MPS per il training
- **Python**: 3.13+
- **ffmpeg**: necessario per WhisperX

```bash
# Installa ffmpeg su macOS
brew install ffmpeg
```

## 📦 Dipendenze

Le dipendenze principali sono gestite in `pyproject.toml`.

| Libreria | Uso | Fase |
|----------|-----|------|
| `mlx` | Framework ML Apple Silicon | 4 |
| `mlx-whisper` | Inferenza Whisper ottimizzata | 4 |
| `wandb` | Tracking training/eval | 4 |
| `jiwer` | Word Error Rate + Medical WER | 4 |
| `librosa` | Feature extraction log-Mel | 4 |
| `pydantic` | Contratti dati SDD | 3.5, 4 |
| `torch` / `torchaudio` | Backend per Fasi 1–3 | 1–3 |
| `soundfile` | Lettura/scrittura WAV | tutte |
| `audiomentations` | Augmentation audio | 3.5 |

### Installazione

```bash
# Installazione con uv (consigliato)
uv sync
```

> **Nota:** tutte le dipendenze sono dichiarate in `pyproject.toml`. `uv sync` installa tutto automaticamente.

## 🚀 Uso

### Preparazione dati

Posiziona i tuoi file nella cartella `data/raw/` come coppie con lo **stesso nome**:

```
data/raw/
├── lezione_anatomia.mp3      # Audio della lezione
├── lezione_anatomia.txt      # Trascrizione manuale completa
├── lezione_patologia.mp3
└── lezione_patologia.txt
```

**Formato trascrizioni**: testo continuo in `.txt`, senza timestamp o formattazione speciale. Esempio:
```
Il paziente presenta una stenosi aortica severa con gradiente 
transvalvolare medio di 45 mmHg e area valvolare calcolata...
```

**Formati audio supportati**: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.wma`

---

### Step 1 — Preparazione Testo

```bash
uv run python scripts/01_prepare_text.py
```

**Cosa fa:**
1. Legge tutti i file `.txt` da `data/raw/` (trascrizioni mediche da PDF)
2. **Pulizia testuale**: rimuove sillabazioni a capo, spazi multipli, artefatti da PDF
3. **Sentence tokenization** con NLTK per la lingua italiana
4. **Hybrid Chunking**: raggruppa le frasi in chunk di 15-35 parole, rispettando il limite di 200 caratteri (vincolo XTTS_v2 per l'italiano). Le frasi troppo lunghe vengono spezzate su virgole e punti e virgola
5. Genera `data/synthetic_chunks/manifest_text.json` con i chunk testuali

**Parametri di chunking** (configurabili nel sorgente):

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `min_words` | `15` | Parole minime per chunk |
| `max_words` | `35` | Parole massime per chunk |
| `max_chars` | `200` | Caratteri massimi (limite XTTS_v2) |

---

### Step 2 — Generazione Audio Sintetico (Coqui XTTS_v2)

```bash
uv run python scripts/02_generate_audio.py
```

> Richiede un file di riferimento vocale `data/raw/reference_voice.wav` per il voice cloning.
> XTTS_v2 supporta solo **CUDA** e **CPU** (MPS non supportato). Su Mac, l'inferenza avviene su CPU.

**Cosa fa:**
1. Carica il manifest testuale da `data/synthetic_chunks/manifest_text.json`
2. Inizializza il modello **Coqui XTTS_v2** (multilingual, voice cloning)
3. Per ogni chunk, genera audio `.wav` con la voce clonata dal riferimento
4. **Resume-safe**: salta automaticamente i chunk già generati (utile per riprendere dopo interruzioni)
5. Salva gli audio in `data/synthetic_audio/` e genera `data/synthetic_audio/manifest_synthetic.json`

**Output:**
- `data/synthetic_audio/*.wav` — Audio sintetici (24kHz, mono)
- `data/synthetic_audio/manifest_synthetic.json` — Manifest con `{id, text, audio_filepath}`

---

### Step 2.5 — Data Augmentation (Synthetic Dataset)

Sviluppata seguendo principi **Spec-Driven Development (SDD)**, questa pipeline permette di irrobustire modelli addestrati su voce sintetica "troppo pulita", iniettando rumori e disturbi tipici di un'aula universitaria (Classroom scenario).

```bash
PYTHONPATH=. uv run python data/augmentation/cli.py data/synthetic_audio/manifest_synthetic.json \
    --intensity 0.7 \
    --p-reverb 0.8 \
    --p-noise 0.9 \
    --p-bandpass 0.3 \
    --p-gain 0.5
```

**Cosa fa:**
1. **Validazione Pydantic**: Controlla rigorosamente che il formato del manifest di input sia corretto e mappa automaticamente alias storici (`audio_filepath` -> `audio_path`).
2. **Logica Probabilistica a Controllo di Degrado**: Esegue una pipeline con 4 tipi di alterazioni indipendenti controllate da flag CLI (riverbero, rumore tastiere/colpi di tosse, banda limitata, variazioni di volume).
3. **Hard Constraint**: Indipendentemente dalle probabilità, non verranno MAI applicati più di **2 filtri contemporaneamente** allo stesso audio. Questo evita di danneggiare troppo l'intelligibilità del discorso.
4. **Resampling sicuro**: Legge i file audio originali a qualsiasi frequenza, ricampionandoli automaticamente alla `sample_rate` voluta (`16000` di default) senza "rallentare" le voci.
5. Esporta un manifesto aggiornato e un batch di audio alterati per procedere alla fase di training.

**Parametri CLI utili:**
- `--intensity` (default: 1.0): Modula l'aggressività delle alterazioni (es. abbassa drasticamente il Signal-to-Noise Ratio).
- `--p-*` (default: 0.5/0.8): Probabilità di attivazione per ogni singolo filtro.

---

### Step 3 — Validazione Dataset
```bash
PYTHONPATH=. uv run python scripts/03_data_contracts.py data/augmented_audio/dataset_augmented.jsonl --verbose
```

### Step 4 — Baseline Benchmark (Zero-Shot)
```bash
PYTHONPATH=. uv run python scripts/05_baseline_benchmark.py \
    --manifest data/augmented_audio/dataset_augmented.jsonl \
    --medical-terms data/medical_terms.txt \
    --output-dir outputs
```

### Step 5 — Preprocessing
```bash
PYTHONPATH=. uv run python scripts/06_preprocess_mlx.py \
    --manifest data/augmented_audio/dataset_augmented.jsonl \
    --output-dir data/preprocessed
```

### Step 6 — Fine-Tuning LoRA
```bash
PYTHONPATH=. uv run python scripts/07_finetune_mlx.py --config training_config.yaml
```

### ⚙️ Configurazione Training

Gli iperparametri sono centralizzati in `training_config.yaml`:

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `model.name` | `mlx-community/whisper-small-mlx` | Modello base |
| `lora.rank` | `32` | Rank LoRA |
| `lora.alpha` | `64` | Alpha LoRA |
| `training.batch_size` | `4` | Batch size (M1 Pro 16GB) |
| `training.learning_rate` | `1e-5` | Learning rate |
| `training.num_epochs` | `3` | Numero epoche |
| `evaluation.eval_every_n_steps` | `50` | Frequenza evaluation |
| `evaluation.medical_weight` | `3.0` | Peso errori termini medici |

### 📊 Weights & Biases Setup

Il training logga automaticamente su [Weights & Biases](https://wandb.ai) per tracciare loss, WER e Medical WER.

#### 1. Login (una tantum)
```bash
wandb login
```
Inserisci la tua API key quando richiesto.

#### 2. Configurazione
Il progetto e l'entity W&B sono configurabili in `training_config.yaml`:
```yaml
wandb:
  project: "whisper-medical-finetuning"
  entity: null  # il tuo username wandb
```

#### 3. Modalità Offline
Per eseguire senza W&B (logging solo su CLI):
```bash
WANDB_MODE=disabled PYTHONPATH=. uv run python scripts/07_finetune_mlx.py --config training_config.yaml
```

Oppure usa il flag `--no-wandb` nello script di benchmark:
```bash
PYTHONPATH=. uv run python scripts/05_baseline_benchmark.py --manifest ... --medical-terms ... --no-wandb
```

## 🔍 Troubleshooting

### WhisperX non si installa
```bash
brew install ffmpeg
pip install git+https://github.com/m-bain/whisperx.git
```

### Errore "MLX out of memory"
- Riduci `training.batch_size` a `2` in `training_config.yaml`
- Riduci `lora.rank` a `16`

### Pochi segmenti dopo il filtro
- Abbassa `--min_speech_ratio` (es. `0.2`)
- Abbassa `--min_snr` (es. `5`)
- Abbassa `--min_similarity` (es. `40`)
