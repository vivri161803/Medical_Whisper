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
- [x] Sviluppo di uno script per la pulizia (text preparation) dei testi raw estratti da PDF.
- [x] Implementazione di un sistema di "hybrid chunking" per dividere il testo ripulito in spezzoni ideali (1-30 secondi).
- [x] Configurazione del modello Text-to-Speech locale `Coqui XTTS_v2`.
- [x] Sviluppo di uno script per generare i file `.wav` e il file `manifest_synthetic.json`.
- [x] Validazione manuale a campione della pronuncia e della qualità del dataset generato.

## Fase 3.5: Data Augmentation
- [x] Definizione dei modelli dati (Pydantic) per la validazione di manifest in ingresso e in uscita.
- [x] Implementazione della pipeline `ClassroomAugmenter` usando `audiomentations` (Reverb, Background Noise, BandPass, Volume Fluctuation).
- [x] Scrittura dei test (pytest) per verificare i requisiti spettrali, dinamici e i vincoli rigidi sulla durata dell'audio (<= 30s).
- [x] Integrazione dello script CLI tramite `Typer` per l'orchestrazione del processing batch.

## Fase 4: Fine-Tuning e Valutazione 
Ecco la roadmap rivista, epurata dalle dipendenze di PyTorch (come `Seq2SeqTrainer` o `torch.utils.data`) e riprogettata nativamente attorno all'ecosistema **Apple MLX**, integrando il rigore dello Spec-Driven Development (SDD) e la fase cruciale di benchmarking.

### A: Spec-Driven Contracts & Baseline (Il punto zero)

Nella cartella scripts:
* [ ] **Sviluppo dello script `03_data_contracts.py`:** Implementazione dei modelli Pydantic per validare la struttura del dataset in ingresso (garantire sample rate a 16kHz, canali mono, durata $\le 30$ secondi e testo ripulito da tag).
* [ ] **Sviluppo dello script `04_metrics.py` (Medical WER):** Creazione di una funzione di valutazione custom che penalizzi maggiormente gli errori sui termini di chirurgia maxillo-facciale rispetto alle congiunzioni o agli articoli, ma anche impostare una loss wer generica come metrica aggiuntiva per supervisionare la fase di training.
* [ ] **Implementazione dello script `05_baseline_benchmark.py`:** Utilizzo di `mlx-whisper` (inferenza pura) per trascrivere un subset di validazione dei tuoi audio aumentati in modalità *zero-shot* (senza addestramento). Registrazione del Medical WER di partenza e WER normale. Questo numero è il benchmark da abbattere.

### B: Data Ingestion Apple-Native

* [ ] **Sviluppo dello script `06_preprocess_mlx.py`:** Estrazione delle feature log-Mel spettrogramma e tokenizzazione del testo. A differenza di PyTorch, calcoleremo queste feature *una tantum* salvandole in array compressi `.npz` (o caricandole via `mlx.data` stream), per eliminare l'overhead della CPU durante il training e scaricare tutto il lavoro sulla GPU dell'M1 Pro.
* [ ] **Gestione dinamica di Padding e Masking (MLX core):** Implementazione di funzioni in `mlx.core` per allineare i tensori audio a lunghezze fisse (es. 3000 frame per Whisper) e mascherare i token di padding nella loss, ignorando i token `-100`.

### C: Configurazione e Training MLX (LoRA)

Sostituiamo il monolitico `Seq2SeqTrainer` con un training loop nativo, leggero e cucito su misura per i chip Apple Silicon tramite il repository `mlx-examples`.

* [ ] **Integrazione del file `training_config.yaml`:** Mappatura centralizzata degli iperparametri essenziali. Configurazione specifica per M1 Pro: `batch_size` ridotto (es. 4 o 8 per non saturare i 16GB), precisione `fp16` (obbligatoria per la velocità), rank LoRA (`r=32`, `alpha=64`), e parametri dell'ottimizzatore (es. AdamW).
* [ ] **Sviluppo dello script `07_finetune_mlx.py` (Il Core):**
* Caricamento dinamico dei pesi originali di `whisper-small` convertiti in formato MLX.
* Iniezione dei moduli LoRA (Low-Rank Adaptation) nei layer di Attention (es. `q_proj`, `v_proj`).
* Congelamento (freezing) del modello base: solo i pesi LoRA avranno `requires_grad = True`.
* [ ] **Sviluppo del MLX Training Loop:** Costruzione del ciclo di addestramento sfruttando `mlx.nn.value_and_grad` per il calcolo della loss e l'aggiornamento dei gradienti tramite `mlx.optimizers`. Gestione esplicita di `mx.eval()` per forzare la valutazione del grafo computazionale in Metal.

### D: Validazione Continua e Checkpointing

* [ ] **Integrazione di W&B (Weights & Biases) o MLX Logging:** Sostituzione di TensorBoard (troppo legato all'ecosistema PyTorch) con un logging leggero su riga di comando o via wandb per tracciare la Loss di training e validation in tempo reale.
* [ ] **Evaluation Loop intra-training:** Configurazione dello script affinché, ogni $N$ step, i pesi LoRA correnti vengano fusi temporaneamente con il modello base per eseguire una trascrizione sul validation set. Calcolo immediato del Medical WER per confermare che l'adattamento al gergo medico stia funzionando.
* [ ] **Salvataggio incrementale degli Adapter (`adapters.npz`):** Checkpointing sicuro che salva *esclusivamente* i pochi megabyte dei pesi LoRA, preservando spazio sul disco del MacBook.

### E: Validazione del codice

Implementare tramite Pytest dei test nella cartella `./tests` per verificare gli script implementati: 

* [ ] Implementare un test per lo script 03 che verifichi che, qualora venga dato un file audio - ad esempio rumore bianco - con le caratteristiche adeguate, questo venga fatto passare dal test
* [ ] Per lo scipt 04 implementare un test che verifichi che le loss possano essere calcolate a partire da dei risultati fasulli, per vedere soltanto se il codice gira. 
* [ ] Per lo script 05 verificare che la parte di inferenza e benchmarking iniziale funzioni
* [ ] Per lo script 06 testare che tutti funzioni a partire da file audio fasulli, per vedere se il codice gira
* [ ] Per lo script 07 testare che le prime iterazioni funzionino e che tuttto parta senza problemi

## Fase 5: Sperimentazione e Deploy (In Pianificazione)
- [ ] Valutazione formale del modello su un test set "hold-out" (lezioni universitarie inedite).
- [ ] Redazione di un report conclusivo del PoC, misurando quantitativamente il delta di accuratezza tra il modello Whisper zero-shot e il modello fine-tunato.
- [ ] (Opzionale) Sperimentazione con architetture di dimensioni superiori (es. `whisper-medium` o `whisper-large-v3-turbo`) se le risorse hardware lo consentono.
- [ ] (Opzionale) Implementazione di uno script CLI dedicato all'inferenza o di una UI semplificata (Gradio/Streamlit) per consentire agli studenti di medicina di trascrivere agilmente le proprie registrazioni.
