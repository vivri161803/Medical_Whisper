# 🏥 Whisper Medical — Fine-Tuning per Terminologia Medica Italiana

Pipeline per il fine-tuning di OpenAI Whisper, specializzata nel riconoscimento del gergo tecnico medico italiano. I modelli open source (come Whisper Medium) tendono a fallire su terminologia clinica, anatomica e farmacologica — questo progetto risolve il problema addestrando il modello su trascrizioni manuali verificate da studenti di medicina.

## 📋 Panoramica

La pipeline si compone di **3 step** sequenziali:

| Step | Script | Descrizione |
|------|--------|-------------|
| 1 | `scripts/01_chunk_and_align.py` | Segmenta audio lunghi e allinea con le trascrizioni manuali |
| 2 | `scripts/02_filter_quality.py` | Filtra segmenti vuoti, silenziosi o troppo rumorosi |
| 3 | `scripts/03_finetune.py` | Fine-tuning di Whisper Small sui segmenti validati |

## 📁 Struttura del Progetto

```
Whisper/
├── scripts/
│   ├── 01_chunk_and_align.py   # Segmentazione + allineamento WhisperX
│   ├── 02_filter_quality.py    # Filtro qualità VAD + SNR
│   └── 03_finetune.py          # Fine-tuning Whisper
├── configs/
│   └── training_config.yaml    # Iperparametri di training
├── data/
│   ├── raw/                    # ⬅️ Metti qui i tuoi audio + .txt
│   ├── chunks/                 # Segmenti generati (Step 1)
│   └── filtered/               # Segmenti filtrati (Step 2)
├── outputs/
│   └── whisper-medical/        # Modello fine-tunato (Step 3)
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

Le seguenti librerie Python sono necessarie:

| Libreria | Versione | Uso |
|----------|----------|-----|
| `whisperx` | latest | Trascrizione + forced alignment |
| `torch` | ≥2.11.0 | Backend ML (già in pyproject.toml) |
| `torchaudio` | ≥2.11.0 | Elaborazione audio (già in pyproject.toml) |
| `transformers` | ≥5.8.0 | Fine-tuning Whisper (già in pyproject.toml) |
| `datasets` | latest | Gestione dataset HuggingFace |
| `accelerate` | latest | Training ottimizzato |
| `evaluate` | latest | Metriche (WER) |
| `jiwer` | latest | Calcolo Word Error Rate |
| `rapidfuzz` | latest | Fuzzy matching testo |
| `silero-vad` | latest | Voice Activity Detection |
| `soundfile` | latest | Lettura/scrittura WAV |
| `librosa` | latest | Analisi audio |
| `pyyaml` | latest | Parsing configurazione |
| `tqdm` | latest | Progress bars |

### Installazione dipendenze

```bash
# Con pip
pip install whisperx rapidfuzz silero-vad soundfile librosa pyyaml tqdm \
            datasets accelerate evaluate jiwer

# Oppure con uv (consigliato)
uv add whisperx rapidfuzz silero-vad soundfile librosa pyyaml tqdm \
       datasets accelerate evaluate jiwer
```

> **Nota:** `torch`, `torchaudio` e `transformers` sono già definiti in `pyproject.toml`.

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

### Step 1 — Segmentazione e Allineamento

```bash
python scripts/01_chunk_and_align.py \
    --input_dir data/raw \
    --output_dir data/chunks \
    --language it \
    --model_size small
```

**Cosa fa:**
1. Trascrive l'audio con WhisperX per ottenere timestamp approssimativi
2. Esegue forced alignment per timestamp word-level precisi
3. Per ogni segmento, cerca la corrispondenza più simile nella trascrizione manuale usando fuzzy matching
4. Sostituisce il testo WhisperX con il testo della trascrizione manuale (il ground truth corretto)
5. Salva i segmenti audio WAV (16kHz mono) in `data/chunks/`
6. Genera `manifest.json` con metadati di ogni segmento

**Parametri opzionali:**

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--input_dir` | `data/raw` | Cartella con audio + .txt |
| `--output_dir` | `data/chunks` | Cartella output segmenti |
| `--language` | `it` | Codice lingua |
| `--model_size` | `small` | Dimensione modello WhisperX |
| `--min_similarity` | `60` | Soglia minima fuzzy matching (%) |
| `--batch_size` | `16` | Batch size per WhisperX |

---

### Step 2 — Filtro Qualità

```bash
python scripts/02_filter_quality.py \
    --input_dir data/chunks \
    --output_dir data/filtered
```

**Cosa fa:**
1. **VAD (Silero)**: Rileva la percentuale di parlato — scarta se < 30%
2. **SNR**: Stima il rapporto segnale/rumore — scarta se < 10 dB
3. **Durata**: Scarta segmenti < 1s o > 30s
4. **Similarity**: Scarta segmenti con allineamento troppo scarso
5. Copia i segmenti che passano tutti i filtri in `data/filtered/`
6. Genera `manifest_filtered.json`

**Parametri opzionali:**

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--min_speech_ratio` | `0.3` | % minima di parlato |
| `--min_snr` | `10.0` | SNR minimo in dB |
| `--min_duration` | `1.0` | Durata minima (secondi) |
| `--max_duration` | `30.0` | Durata massima (secondi) |
| `--min_similarity` | `50.0` | Similarity score minimo (%) |

---

### Step 3 — Fine-Tuning

```bash
python scripts/03_finetune.py \
    --data_dir data/filtered \
    --config configs/training_config.yaml
```

**Cosa fa:**
1. Carica `openai/whisper-small` e il processor per italiano
2. Crea dataset PyTorch dai segmenti filtrati (log-Mel spectrograms + labels)
3. Split automatico 90/10 train/validation
4. Allena il modello con `Seq2SeqTrainer`
5. Valuta con WER (Word Error Rate) ad ogni epoca
6. Salva il miglior modello in `outputs/whisper-medical/final/`

**Monitoraggio training con TensorBoard:**
```bash
tensorboard --logdir outputs/whisper-medical
```

### Configurazione Training

Gli iperparametri sono in `configs/training_config.yaml`. I principali:

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `model_name` | `openai/whisper-small` | Modello base |
| `learning_rate` | `1e-5` | Learning rate |
| `per_device_train_batch_size` | `4` | Batch size per device |
| `gradient_accumulation_steps` | `4` | Effective batch = 16 |
| `num_train_epochs` | `10` | Numero epoche |
| `fp16` | `false` | Disabilitato per MPS |

> Per usare un modello più grande su GPU NVIDIA, cambia `model_name` a `openai/whisper-medium`, abilita `fp16: true`, e aumenta il batch size.

## 🔍 Troubleshooting

### WhisperX non si installa
```bash
# Assicurati di avere ffmpeg
ffmpeg -version

# Installa da source se pip fallisce
pip install git+https://github.com/m-bain/whisperx.git
```

### Errore "MPS out of memory"
- Riduci `per_device_train_batch_size` a `2` nel config YAML
- Riduci `--batch_size` nello Step 1

### Training troppo lento su CPU
- Assicurati che PyTorch sia compilato con supporto MPS:
```python
import torch
print(torch.backends.mps.is_available())  # Deve essere True
```

### Pochi segmenti dopo il filtro
- Abbassa `--min_speech_ratio` (es. `0.2`)
- Abbassa `--min_snr` (es. `5`)
- Abbassa `--min_similarity` (es. `40`)
