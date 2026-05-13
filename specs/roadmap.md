# Roadmap

La roadmap è organizzata in fasi di lavoro logiche e sequenziali, rispecchiando l'attuale struttura della pipeline del progetto. Essendo un Proof of Concept, il focus primario è sulla stabilità della preparazione del dato e sulla riproducibilità del training.

## Fase 1: Segmentazione e Allineamento (Completato)
- [x] Sviluppo dello script `01_chunk_and_align.py`.
- [x] Trascrizione iniziale degli audio lunghi e forced alignment tramite WhisperX.
- [x] Implementazione dell'algoritmo di fuzzy matching (RapidFuzz) per sostituire i testi allineati automaticamente con il ground truth della trascrizione manuale verificata.
- [x] Taglio e generazione di segmenti audio brevi (1-30 secondi) salvati in `data/chunks/` con il rispettivo file `manifest.json`.

## Fase 2: Filtro Qualità (Completato)
- [x] Sviluppo dello script `02_filter_quality.py`.
- [x] Rilevamento della percentuale di parlato per segmento tramite Silero VAD (es. >30% di parlato).
- [x] Stima euristica del Signal-to-Noise Ratio (SNR) per scartare segmenti eccessivamente rumorosi (<10 dB).
- [x] Filtraggio secondario basato su anomalie di durata e score di similarity insufficiente con il ground truth.
- [x] Salvataggio del dataset audio purificato in `data/filtered/` con relativo `manifest_filtered.json`.

## Fase 3 (Pivot): Generazione Dataset Audio Sintetico
- [ ] Sviluppo di uno script per la pulizia (text preparation) dei testi raw estratti da PDF.
- [ ] Implementazione di un sistema di "hybrid chunking" per dividere il testo ripulito in spezzoni ideali (1-30 secondi).
- [ ] Configurazione del modello Text-to-Speech locale `Coqui XTTS_v2`.
- [ ] Sviluppo di uno script per generare i file `.wav` e il file `manifest_synthetic.json`.
- [ ] Validazione manuale a campione della pronuncia e della qualità del dataset generato.

## Fase 4: Fine-Tuning e Valutazione 
- [ ] Sviluppo dello script di training `03_finetune.py`.
- [ ] Sviluppo di un dataset custom in PyTorch per il caricamento e l'estrazione delle log-Mel feature dal dataset sintetico.
- [ ] Implementazione del data collator per la gestione del padding dinamico e del masking delle label.
- [ ] Configurazione del `Seq2SeqTrainer` per orchestrare l'addestramento ottimizzando sulle risorse hardware disponibili (Apple Silicon MPS).
- [ ] Integrazione del file `training_config.yaml` per la gestione centralizzata degli iperparametri.
- [ ] Monitoraggio delle loss via TensorBoard e valutazione periodica delle prestazioni tramite la metrica Word Error Rate (WER).

## Fase 5: Sperimentazione e Deploy (In Pianificazione)
- [ ] Valutazione formale del modello su un test set "hold-out" (lezioni universitarie inedite).
- [ ] Redazione di un report conclusivo del PoC, misurando quantitativamente il delta di accuratezza tra il modello Whisper zero-shot e il modello fine-tunato.
- [ ] (Opzionale) Sperimentazione con architetture di dimensioni superiori (es. `whisper-medium` o `whisper-large-v3-turbo`) se le risorse hardware lo consentono.
- [ ] (Opzionale) Implementazione di uno script CLI dedicato all'inferenza o di una UI semplificata (Gradio/Streamlit) per consentire agli studenti di medicina di trascrivere agilmente le proprie registrazioni.
