# Implementation Plan: Synthetic Audio Dataset Generation

## Task Group 1: Text Preparation and Cleaning
- [ ] Create a script `01_clean_pdf_text.py` to read `.txt` files from `data/raw`.
- [ ] Implement regex/heuristics to remove PDF artifacts: page numbers, headers, footers, newlines mid-sentence, and hyphenated line breaks.
- [ ] Output the cleaned, continuous text into an intermediate file or variable.

## Task Group 2: Hybrid Text Chunking
- [ ] Create a script `02_chunk_text.py` (può essere unito al precedente).
- [ ] Implement a hybrid chunking algorithm: first split by sentences using punctuation, then group or split sentences to ensure each chunk represents approximately 5-25 seconds of speech (roughly 10-50 words).
- [ ] Generate a JSON/CSV manifest containing the chunks with their corresponding IDs.

## Task Group 3: Synthetic Audio Generation
- [ ] Create a script `03_generate_audio_xtts.py`.
- [ ] Setup the local Coqui XTTS_v2 model.
- [ ] Iterate through the chunked manifest, feeding each text chunk to XTTS_v2.
- [ ] Save the generated audio files as `.wav` in a new `data/synthetic_audio` folder.
- [ ] Create a final `manifest_synthetic.json` mapping each audio file to its text, perfectly formatted and ready for Whisper fine-tuning.
