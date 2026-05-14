# Tech Stack

L'infrastruttura tecnologica del progetto si concentra sull'ecosistema Python per il Machine Learning e l'elaborazione audio, con focus sull'ottimizzazione nativa per **Apple Silicon** tramite l'ecosistema **MLX**.

## Linguaggio e Core
- **Python 3.11+**: Linguaggio di programmazione principale per l'intera pipeline (downgrade per compatibilità con librerie TTS).
- **uv / pip**: Gestione delle dipendenze e degli ambienti virtuali.

## Machine Learning & Deep Learning (MLX-Native)
- **MLX (≥0.31.2)**: Framework ML nativo per Apple Silicon. Utilizzato per l'intero training loop (LoRA fine-tuning) con `mlx.nn`, `mlx.optimizers` e `mlx.core`.
- **MLX-Whisper (≥0.4.3)**: Inferenza Whisper ottimizzata per Apple Silicon. Utilizzato per il baseline benchmark e l'evaluation intra-training.
- **Weights & Biases (≥0.27.0)**: Tracking della telemetria di training (loss, WER, Medical WER) in tempo reale.

## Machine Learning Legacy (Fasi 1–3)
- **PyTorch (≥2.11.0)**: Backend per Silero VAD e WhisperX (Fasi 1–2).
- **Transformers (HuggingFace)**: Utilizzato esclusivamente per WhisperX (Fase 1).
- **Datasets & Evaluate (HuggingFace)**: Gestione efficiente del dataset in memoria.
- ~~**Accelerate**~~: *Deprecato — sostituito dal training loop MLX-nativo.*
- **WhisperX**: Trascrizione iniziale e forced alignment word-level rapido ed accurato.
- **Silero VAD**: Modello pre-addestrato per il rilevamento dell'attività vocale (Voice Activity Detection).
- **Coqui TTS (XTTS_v2)**: Generazione audio sintetica ad alta qualità (Text-to-Speech) in esecuzione locale.

## Elaborazione Audio e Dati
- **Torchaudio**: Elaborazione nativa e resampling dei tensori audio.
- **Soundfile & Librosa**: Lettura, scrittura e feature extraction di file audio in vari formati.
- **FFmpeg**: Backend di sistema essenziale per il decoding audio (dipendenza bloccante per WhisperX).

## Utility
- **RapidFuzz**: Fuzzy string matching veloce per allineare le stringhe riconosciute da WhisperX al ground truth della trascrizione manuale.
- **Jiwer**: Calcolo efficiente del Word Error Rate (WER) per valutare metricamente le prestazioni del modello ad ogni epoca.
- **Tqdm**: Barre di progresso visive per monitorare l'avanzamento dei task di processamento sequenziale.
- **PyYAML**: Lettura e gestione dei file di configurazione dichiarativa (`training_config.yaml`).
- ~~**TensorBoard**~~: *Deprecato — sostituito da Weights & Biases (wandb).*

## Validazione, Testing e Data Augmentation
- **Pydantic**: Validazione rigorosa e definizione dei contratti dei dati per l'implementazione SDD (Spec-Driven Development).
- **Pytest**: Framework per la scrittura e l'esecuzione di test funzionali e validazione delle specifiche.
- **Audiomentations**: Creazione di pipeline componibili per perturbare e simulare condizioni di disturbo sul segnale audio.
- **Typer**: Sviluppo di interfacce CLI (Command Line Interface) tipizzate e robuste per l'orchestrazione dei batch.
