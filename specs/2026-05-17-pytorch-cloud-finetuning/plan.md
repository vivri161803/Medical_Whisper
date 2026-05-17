# Plan — PyTorch Cloud Fine-Tuning Pipeline

## Task Group 1: Infrastruttura (`scripts_cloud/`)

1.1. Creare la directory `scripts_cloud/` con struttura:
```
scripts_cloud/
├── training_config.yaml    # Config PyTorch-specifica
├── requirements.txt        # Dipendenze per RunPod (PyTorch 2.4.0)
├── finetune_pytorch.py     # Training loop principale
├── dataset.py              # Custom Dataset per .npz
├── metrics.py              # WER + Medical WER (portato da scripts/)
└── deploy_runpod.sh        # Script bash per setup con uv + lancio su RunPod
```

1.2. Creare `requirements.txt` con PyTorch 2.4.0 + CUDA, transformers, peft, datasets, jiwer, wandb, tqdm, pyyaml, safetensors.

1.3. Creare `training_config.yaml` con parametri adattati per PyTorch/L4.

## Task Group 2: Dataset Loader (`dataset.py`)

2.1. Implementare `WhisperNpzDataset(torch.utils.data.Dataset)` che:
- Carica file `.npz` da una directory
- Restituisce `log_mel` come `torch.FloatTensor` trasposto a `(80, 3000)` (formato PyTorch Whisper: channels-first)
- Restituisce `labels` come `torch.LongTensor`
- Supporta padding/collation via custom `collate_fn`

2.2. Implementare `DataCollatorForWhisperNpz` per gestire il batching con padding dei labels.

## Task Group 3: Metriche (`metrics.py`)

3.1. Portare `compute_wer` e `compute_medical_wer` dal modulo `scripts/metrics.py`.
3.2. Adattare `load_medical_terms` per funzionare standalone.
3.3. Implementare `compute_metrics_fn` compatibile con `Seq2SeqTrainer`.

## Task Group 4: Training Script (`finetune_pytorch.py`)

4.1. Caricamento modello via `WhisperForConditionalGeneration.from_pretrained()`.
4.2. Applicazione LoRA via `peft.LoraConfig` + `get_peft_model`:
- `target_modules=["q_proj", "v_proj"]`
- `task_type=TaskType.SEQ_2_SEQ_LM`
4.3. Setup `Seq2SeqTrainingArguments` con:
- fp16, gradient accumulation, warmup, logging
- Early stopping via `EarlyStoppingCallback` sulla metrica wer
4.4. Training via `Seq2SeqTrainer` con:
- Custom dataset
- Custom compute_metrics
- W&B logging
4.5. Salvataggio adapter LoRA finali via `model.save_pretrained()`.

## Task Group 5: Deploy Script (`deploy_runpod.sh`)

5.1. Script bash per RunPod che:
- Installa dipendenze da `requirements.txt`
- Sincronizza i dati preprocessati
- Lancia il training
- Salva i risultati
