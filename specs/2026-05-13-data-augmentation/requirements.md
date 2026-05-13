# Requirements: Data Augmentation Pipeline

## Scopo e Contesto
Creare una pipeline di data augmentation componibile per perturbare dataset di audio sintetico o pulito, simulando rumori tipici di un'aula universitaria (riverbero, rumore di fondo, limiti hardware e variazioni dinamiche). L'obiettivo è esporre il modello Whisper ad una maggiore varianza di input per renderlo più robusto durante il fine-tuning sul gergo medico italiano.

## Specifiche Tecniche
1. **Contratti Dati (I/O)**:
   - Ingressi validati rigorosamente tramite Pydantic (`AudioTextPair`).
   - Uscite validate tramite Pydantic (`AugmentedAudioTextPair`).
   - Vincolo fondamentale: nessuna traccia audio in ingresso o in uscita deve superare i 30.0 secondi.

2. **Trasformazioni (Pipeline `ClassroomAugmenter`)**:
   - *Reverb*: Utilizzo di Impulse Response (IR) files (es. `data/augmentation/ir`). Il riverbero non deve allungare l'audio oltre i 30s.
   - *Background Noise*: Mix con rumori di sottofondo ambientali (es. `data/augmentation/backnoise`). L'SNR (Signal-to-Noise Ratio) deve variare casualmente tra +5dB e +15dB.
   - *Hardware Limit (BandPass)*: Frequenza di taglio inferiore a 300Hz e superiore a 6000Hz per simulare microfoni scadenti o limitazioni del canale.
   - *Volume Fluctuation*: Applica un gain dinamico che oscilla lentamente del $\pm 15\%$ lungo i 30 secondi.

3. **Infrastruttura**:
   - `audiomentations` per la pipeline audio componibile.
   - `torchaudio` o `soundfile` per lettura e scrittura efficiente su disco.
   - `Typer` per esporre i parametri e avviare il batch job via CLI.
   - Orchestrazione con probabilità di applicazione configurabile (es. $p=0.8$) per garantire esempi puliti alternati a quelli degradati.
