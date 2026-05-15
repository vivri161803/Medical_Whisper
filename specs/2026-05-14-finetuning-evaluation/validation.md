# Validation: Fase 4 — Fine-Tuning e Valutazione

L'implementazione si intende validata e pronta per il merge quando **tutti** i seguenti smoke test passano tramite `pytest`. I test sono progettati per essere veloci (< 30s ciascuno) e non richiedere hardware GPU specifico.

---

## 1. Validazione Contratti Dati (`test_data_contracts.py`)

- **Azione:** Generare un file WAV di rumore bianco (16kHz, mono, 5s) con `numpy` + `soundfile`. Creare un manifest JSON contenente il riferimento a questo file.
- **Risultato Atteso:** `validate_manifest()` restituisce un `ValidationReport` con status `pass` per questo file.
- **Caso Negativo:** Fornire un file audio stereo a 48kHz → il validatore deve rifiutarlo con errore specifico su sample rate e canali.

## 2. Validazione Metriche (`test_metrics.py`)

- **Azione:** Creare due coppie reference/hypothesis fasulle:
  - Coppia A: errore su una congiunzione ("e" → "o").
  - Coppia B: errore su un termine medico ("mandibola" → "mandibula").
- **Risultato Atteso:**
  - `compute_wer()` restituisce un float > 0 per entrambe le coppie.
  - `compute_medical_wer()` restituisce un valore **maggiore o uguale** al WER standard per la Coppia B (errore medico penalizzato).
  - Entrambe le funzioni non sollevano eccezioni.

## 3. Validazione Baseline Benchmark (`test_baseline_benchmark.py`)

- **Azione:** Generare un audio sintetico minimo (sine wave 1s, 16kHz) e un manifest con ground truth fittizio. Eseguire l'inferenza zero-shot tramite `mlx_whisper.transcribe()`.
- **Risultato Atteso:**
  - Lo script produce un report JSON valido con i campi `wer`, `medical_wer`, `file_results`.
  - L'inferenza non solleva eccezioni (la qualità della trascrizione non è verificata).
  - Il report JSON è deserializzabile e contiene valori numerici finiti.

## 4. Validazione Preprocessing (`test_preprocess_mlx.py`)

- **Azione:** Generare 5 file audio sintetici (rumore bianco 3s, 16kHz) con testi fittizi. Eseguire `06_preprocess_mlx.py` con split 80/10/10.
- **Risultato Atteso:**
  - Le feature log-Mel hanno shape `(N, 80)` dove N ≤ 3000 (formato nativo mlx-whisper: `n_frames, n_mels`).
  - Dopo padding, la dimensione è esattamente `(3000, 80)`.
  - I token di padding nel label sono `-100`.
  - I file `.npz` vengono creati nelle 3 sottocartelle (`train/`, `val/`, `test/`).
  - Le 3 sottocartelle sono non vuote (almeno 1 file ciascuna con 5 sample e seed fisso).

## 5. Validazione Fine-Tuning LoRA (`test_finetune_mlx.py`)

- **Azione:** Creare un modello mock minimale (1 layer di self-attention con `query` e `value`). Applicare LoRA con rank=4. Eseguire 2 iterazioni di training con batch di dati randomici.
- **Risultato Atteso:**
  - Dopo `model.freeze()` + applicazione LoRA, solo i parametri LoRA risultano trainabili.
  - Il rapporto `trainable / total` parametri è < 5%.
  - Dopo 2 step di training, la loss non è `NaN` né `Inf`.
  - Il file `adapters.npz` viene salvato con successo.
  - Il file `adapters.npz` è ricaricabile e contiene le chiavi dei parametri LoRA.

## 6. Validazione WER Evaluation (`test_finetune_mlx.py::TestWEREvaluation`)

- **Azione:** Caricare il modello Whisper reale e i dati di validazione preprocessati. Eseguire la trascrizione autoregressiva e calcolare le metriche.
- **Risultato Atteso:**
  - Le mel features sono in formato nativo `(3000, 80)` con scala `~[-1, 2]`.
  - I token di riferimento decodificano in testo non vuoto.
  - `whisper_decode` con `fp16=False` produce testo valido.
  - `whisper_decode` con `fp16=True` lancia `TypeError` (regression guard).
  - La WER su singolo campione è < 1.0.
  - `compute_epoch_wer` restituisce `eval/wer` e `eval/medical_wer` entrambi < 1.0.

---

## Esecuzione dei Test

Tutti i test possono essere eseguiti con:

```zsh
PYTHONPATH=. uv run pytest tests/test_data_contracts.py tests/test_metrics.py tests/test_baseline_benchmark.py tests/test_preprocess_mlx.py tests/test_finetune_mlx.py -v
```

Oppure singolarmente:

```zsh
# Gruppo 1
PYTHONPATH=. uv run pytest tests/test_data_contracts.py tests/test_metrics.py -v

# Gruppo 2
PYTHONPATH=. uv run pytest tests/test_baseline_benchmark.py -v

# Gruppo 3
PYTHONPATH=. uv run pytest tests/test_preprocess_mlx.py -v

# Gruppo 4
PYTHONPATH=. uv run pytest tests/test_finetune_mlx.py -v
```

## Criterio di Merge

Il branch `feat/fase4-finetuning-evaluation` può essere mergiato nel main quando:

1. ✅ Tutti e 5 i gruppi di test passano (`pytest` exit code 0).
2. ✅ Il file `training_config.yaml` è presente e parsabile.
3. ✅ Il `README.md` contiene le istruzioni per W&B setup.
4. ✅ Il `pyproject.toml` non contiene più le dipendenze deprecate.
5. ✅ Lo script `05_baseline_benchmark.py` ha prodotto almeno un `baseline_report.json` valido su dati reali.
