# Requirements: Synthetic Audio Dataset Generation

## Scope
The goal of this phase is to pivot from fine-tuning on noisy, real-world audio to generating a clean, high-quality synthetic dataset using Text-to-Speech (TTS). The source text comes from PDF copy-pastes containing medical transcripts, currently located in `data/raw`.

## Decisions & Context
1. **Text Preparation and Cleaning**: The input `.txt` files are copy-pasted from PDFs. They require cleaning to remove formatting artifacts (e.g., broken lines, hyphenated words across lines) to ensure the best possible audio output. Implemented via regex in `scripts/01_prepare_text.py`.
2. **Text Chunking**: A hybrid chunking strategy is implemented in the same script (`01_prepare_text.py`). It splits cleaned text by sentences (NLTK italian tokenizer), then groups or splits to respect: `min_words=15`, `max_words=35`, `max_chars=200` (XTTS_v2 character limit for Italian).
3. **Audio Generation (TTS)**: We use **Coqui XTTS_v2** running locally (`scripts/02_generate_audio.py`) to synthesize audio from chunked text with voice cloning from a reference speaker. The script is resume-safe (skips already generated files).
4. **Validation**: The validation is strictly manual — the user spot-checks the generated audio samples.

## Implementation Notes
- The original plan proposed 3 separate scripts (`01_clean`, `02_chunk`, `03_generate`). In practice, text cleaning and chunking were unified into `01_prepare_text.py` as the plan suggested ("può essere unito al precedente"), and audio generation became `02_generate_audio.py`.
- XTTS_v2 requires a `weights_only=False` patch for PyTorch 2.6+ compatibility.
- XTTS_v2 supports only CUDA and CPU (MPS not supported). On Mac, inference runs on CPU.

## Tech Stack
- `TTS` (Coqui) per la generazione audio in locale con il modello `XTTS_v2`.
- `nltk` per la sentence tokenization (`sent_tokenize`, lingua italiana).
- Regex per la pulizia del testo da artefatti PDF.
