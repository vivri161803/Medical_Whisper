# Requirements: Synthetic Audio Dataset Generation

## Scope
The goal of this phase is to pivot from fine-tuning on noisy, real-world audio to generating a clean, high-quality synthetic dataset using Text-to-Speech (TTS). The source text comes from PDF copy-pastes containing medical transcripts, currently located in `data/raw`.

## Decisions & Context
1. **Text Preparation and Cleaning**: The input `.txt` files are copy-pasted from PDFs. They require cleaning to remove formatting artifacts (e.g., page numbers, header/footer, broken lines, hyphenated words across lines) to ensure the best possible audio output.
2. **Text Chunking**: We will implement a hybrid chunking strategy. It must intelligently split the cleaned text (e.g., by punctuation/sentences) while respecting the optimal length constraints for Whisper training (audio segments of 1 to 30 seconds). 
3. **Audio Generation (TTS)**: We will use **Coqui XTTS_v2** running locally to synthesize the audio from the chunked text. This ensures high-quality voice generation and perfect alignment between the text and the resulting audio without relying on external APIs.
4. **Validation**: The validation will be strictly manual to avoid overcomplicating the setup. The user will spot-check the generated audio samples.

## Tech Stack Additions
- `TTS` (Coqui) per la generazione audio in locale con il modello `XTTS_v2`.
- Librerie standard o regex-based (e.g., `nltk` o `spacy`) per la pulizia del testo da artefatti PDF e per il sentence splitting.
