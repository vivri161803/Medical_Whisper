# Tech Stack

L'infrastruttura tecnologica del progetto si concentra sull'ecosistema Python per il Machine Learning e l'elaborazione audio, con focus sull'ottimizzazione per acceleratori hardware come Apple Silicon (MPS).

## Linguaggio e Core
- **Python 3.11+**: Linguaggio di programmazione principale per l'intera pipeline (downgrade per compatibilità con librerie TTS).
- **uv / pip**: Gestione delle dipendenze e degli ambienti virtuali.

## Machine Learning & Deep Learning
- **PyTorch (≥2.11.0)**: Backend principale per il machine learning, configurato per utilizzare l'accelerazione hardware Apple Silicon (`mps`) e `cuda` dove disponibile.
- **Transformers (HuggingFace, ≥5.8.0)**: Gestione del modello Whisper, del processor e utilizzo del `Seq2SeqTrainer` per il loop di fine-tuning.
- **Datasets & Evaluate (HuggingFace)**: Gestione efficiente del dataset in memoria e calcolo standardizzato delle metriche di valutazione.
- **Accelerate**: Per l'ottimizzazione avanzata e la scalabilità del training loop.
- **WhisperX**: Trascrizione iniziale e forced alignment word-level rapido ed accurato.
- **Silero VAD**: Modello pre-addestrato per il rilevamento dell'attività vocale (Voice Activity Detection) ad altissima precisione.
- **Coqui TTS (XTTS_v2)**: Generazione audio sintetica ad alta qualità (Text-to-Speech) in esecuzione locale, per la creazione di un dataset pulito.

## Elaborazione Audio e Dati
- **Torchaudio**: Elaborazione nativa e resampling dei tensori audio.
- **Soundfile & Librosa**: Lettura, scrittura e feature extraction di file audio in vari formati.
- **FFmpeg**: Backend di sistema essenziale per il decoding audio (dipendenza bloccante per WhisperX).

## Utility
- **RapidFuzz**: Fuzzy string matching veloce per allineare le stringhe riconosciute da WhisperX al ground truth della trascrizione manuale.
- **Jiwer**: Calcolo efficiente del Word Error Rate (WER) per valutare metricamente le prestazioni del modello ad ogni epoca.
- **Tqdm**: Barre di progresso visive per monitorare l'avanzamento dei task di processamento sequenziale.
- **PyYAML**: Lettura e gestione dei file di configurazione dichiarativa (`training_config.yaml`).
- **TensorBoard**: Tracciamento dei log di addestramento e monitoraggio delle metriche (loss, WER) nel tempo.

## Validazione, Testing e Data Augmentation
- **Pydantic**: Validazione rigorosa e definizione dei contratti dei dati per l'implementazione SDD (Spec-Driven Development).
- **Pytest**: Framework per la scrittura e l'esecuzione di test funzionali e validazione delle specifiche.
- **Audiomentations**: Creazione di pipeline componibili per perturbare e simulare condizioni di disturbo sul segnale audio.
- **Typer**: Sviluppo di interfacce CLI (Command Line Interface) tipizzate e robuste per l'orchestrazione dei batch.
