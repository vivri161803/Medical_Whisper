# Implementation Plan: Data Augmentation Pipeline

Questo documento definisce il piano di implementazione iterativo (TDD/SDD) per la pipeline di data augmentation.

## 1. Definizione dei Contratti (Pydantic)
- [x] Creare `data/augmentation/models.py`.
- [ ] Implementare `AudioTextPair` e `AugmentedAudioTextPair` secondo le specifiche.
- [x] Scrivere un test di unità per validare un manifesto JSON (positivo e negativo) con Pydantic.

## 2. Configurazione e Specs della Pipeline di Augmentation
- [x] Creare `tests/test_augmentation.py` per le specifiche.
- [x] Scrivere Spec per il Bandpass (controllo spettro tramite `librosa`).
- [x] Scrivere Spec per la Fluttuazione Volume (controllo RMSE).
- [x] Scrivere Spec per la Durata (controllo padding/trimming a max 30.0s).
- [x] Creare `data/augmentation/pipeline.py` con la classe `ClassroomAugmenter` usando `audiomentations`.
- [x] Implementare i 4 filtri: Reverb, Background Noise, BandPass e Gain Fluctuation.
- [x] Far passare tutti i test delle specifiche.

## 3. Workflow ed Orchestrazione CLI
- [x] Creare `data/augmentation/cli.py` usando `Typer`.
- [x] Implementare la logica di ingestion: lettura e validazione di `dataset_clean.json`.
- [x] Implementare il batch processing: caricamento audio in array/tensori e passaggio tramite `ClassroomAugmenter`.
- [x] Implementare l'export: salvataggio nuovi `.wav` e scrittura del manifesto in formato `dataset_augmented.jsonl` validato con Pydantic.
- [x] Test end-to-end su un mini-batch di file reali o mock.

Per utilizzare la CLI, eccone un esempio:

```zsh
PYTHONPATH=. uv run python data/augmentation/cli.py data/synthetic_audio/manifest_synthetic.json \
  --intensity 0.7 \
  --p-reverb 0.8 \
  --p-noise 0.9 \
  --p-bandpass 0.3 \
  --p-gain 0.5
```

Per capire al meglio come possa essere utilizzata dare un'occhiata al file: [text](../../data/augmentation/cli.py)