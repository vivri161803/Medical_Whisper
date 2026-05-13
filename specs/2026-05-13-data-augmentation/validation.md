# Validation: Data Augmentation Pipeline

La pipeline si intende validata ed implementabile quando le seguenti condizioni sono soddisfatte automaticamente tramite test `pytest`:

## 1. Validazione Spettrale (BandPass)
- **Azione:** Fornire un file audio di rumore bianco.
- **Risultato Atteso:** L'analisi spettrale effettuata tramite `librosa` sull'output deve confermare che l'energia sotto i 300Hz e sopra i 6000Hz è vicina o uguale a zero.

## 2. Validazione Fluttuazione (Dynamic Gain)
- **Azione:** Fornire un'onda sinusoidale perfetta a volume costante.
- **Risultato Atteso:** L'analisi della Root Mean Square (RMS) dell'output deve evidenziare una variazione "dolce", dimostrando la fluttuazione del gain ($\pm 15\%$).

## 3. Validazione Vincolo Durata (Trim/Pad)
- **Azione:** Fornire un audio pulito di 29.9 secondi e applicare un effetto di riverbero prolungato.
- **Risultato Atteso:** La lunghezza totale del tensore/array di output non deve eccedere la durata di 30.0 secondi (viene troncata la coda o riempita tramite padding se necessario).

## 4. Validazione Contratti Pydantic
- **Azione:** Fornire a Pydantic un manifest in ingresso/uscita con durata > 30.0s o file audio inesistente.
- **Risultato Atteso:** Viene lanciata un'eccezione di validazione Pydantic prima che qualsiasi elaborazione audio abbia inizio.

I test possono essere utilizzati da terminale con:

```zsh
PYTHONPATH=. uv run pytest tests/test_augmentation.py 
```