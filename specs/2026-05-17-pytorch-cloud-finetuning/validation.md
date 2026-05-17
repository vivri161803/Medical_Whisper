# Validation — PyTorch Cloud Fine-Tuning Pipeline

## V1: Dataset Loader
- [ ] `WhisperNpzDataset` carica correttamente i file `.npz` dal filesystem
- [ ] Le shape sono corrette: `log_mel → (80, 3000)`, `labels → (448,)`
- [ ] Il collator produce batch con padding corretto e attention mask

## V2: Metriche
- [ ] `compute_wer` produce risultati identici al modulo MLX
- [ ] `compute_medical_wer` penalizza correttamente i termini medici
- [ ] La funzione `compute_metrics` è compatibile con il Trainer

## V3: Training Loop
- [ ] Il modello si carica correttamente (whisper-small, medium, large-v3)
- [ ] LoRA viene applicato ai moduli corretti (`q_proj`, `v_proj`)
- [ ] Il training gira senza errori OOM su L4 24GB con whisper-medium batch_size=4
- [ ] Early stopping si attiva dopo N epoche senza miglioramento
- [ ] I checkpoint LoRA vengono salvati in formato PEFT-compatibile
- [ ] W&B riceve i log di training e validazione

## V4: Deploy
- [ ] `deploy_runpod.sh` installa l'environment correttamente
- [ ] Il training completo di whisper-medium termina entro 5h su L4
- [ ] I risultati (adapter + report) sono recuperabili dopo il training

## V5: Integrità cross-pipeline
- [ ] Gli adapter PyTorch possono essere riconvertiti per inferenza locale
- [ ] Il Medical WER del modello PyTorch-trained è confrontabile con il modello MLX-trained
