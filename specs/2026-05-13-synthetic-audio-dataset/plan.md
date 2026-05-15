# Implementation Plan: Synthetic Audio Dataset Generation

## Task Group 1: Text Preparation and Cleaning + Chunking

> **Nota:** I task group 1 e 2 originali sono stati unificati in un singolo script come suggerito nel plan iniziale ("può essere unito al precedente").

### Script `01_prepare_text.py`
- [x] Leggere tutti i file `.txt` da `data/raw/`.
- [x] Implementare la pulizia con regex: rimozione sillabazioni a capo (`-\n`), newline mid-sentence, spazi multipli.
- [x] Implementare sentence tokenization con `nltk.sent_tokenize(language='italian')`.
- [x] Implementare Hybrid Chunking con vincoli:
  - `min_words=15`, `max_words=35` (range ottimale per Whisper ~5-25 secondi).
  - `max_chars=200` (limite XTTS_v2 per la lingua italiana).
  - Split delle frasi troppo lunghe su virgole e punti e virgola.
- [x] Generare `data/synthetic_chunks/manifest_text.json` con `{id, text, source_file}`.

## ~~Task Group 2: Hybrid Text Chunking~~ (Unificato nel Task Group 1)

## Task Group 3: Synthetic Audio Generation

### Script `02_generate_audio.py`
- [x] Configurare il modello Coqui XTTS_v2 locale (con patch `weights_only=False` per PyTorch 2.6+).
- [x] Caricare il manifest testuale da `data/synthetic_chunks/manifest_text.json`.
- [x] Implementare voice cloning da file di riferimento (`data/raw/reference_voice.wav`).
- [x] Iterare sui chunk, generando file `.wav` in `data/synthetic_audio/`.
- [x] Implementare resume-safe: skip automatico dei chunk già generati.
- [x] Generare il manifest finale `data/synthetic_audio/manifest_synthetic.json` con `{id, text, audio_filepath}`.
- [x] Validazione manuale a campione della pronuncia e della qualità audio.
