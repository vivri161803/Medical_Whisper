# Requirements — PyTorch Cloud Fine-Tuning Pipeline

## Scope

Porting del training loop da MLX-native a PyTorch + HuggingFace + PEFT per esecuzione su GPU NVIDIA cloud (RunPod L4 24GB VRAM). Il codice vive in `scripts_cloud/` ed è autonomo rispetto al codice MLX esistente.

## Context

- Il fine-tuning MLX su M1 Pro 16GB è limitato a whisper-small (~2.5h/epoca).
- whisper-medium (769M params) richiede più memoria di quanto l'M1 Pro possa gestire con batch_size ragionevoli.
- RunPod offre GPU L4 (24GB VRAM) on-demand, adatte al fine-tuning LoRA di whisper-medium.
- I dati preprocessati esistono già come file `.npz` con formato: `log_mel (3000, 80) float32`, `labels (448,) int32`, `text str`.

## Decisions

### D1: Riutilizzo dati preprocessati
I file `.npz` (mel + labels + text) generati da `06_preprocess_mlx.py` sono il **formato canonico di input**. Lo script PyTorch li carica direttamente via un custom `torch.utils.data.Dataset`.

### D2: Config separata
File `scripts_cloud/training_config.yaml` dedicato, adattato per PyTorch (es. `fp16: true`, `gradient_accumulation_steps`, nomi modello HuggingFace).

### D3: PEFT/LoRA standard
Utilizzo della libreria `peft` di HuggingFace con `LoraConfig` + `get_peft_model`. Target modules: `q_proj`, `v_proj` (naming PyTorch di Whisper).

### D4: HuggingFace Seq2SeqTrainer
Uso del `Seq2SeqTrainer` per il training loop, con callbacks custom per Medical WER e early stopping.

### D5: Target hardware
RunPod L4 (24GB VRAM), PyTorch 2.4.0, CUDA. Requirements.txt autocontenuto.

### D6: Modelli supportati
Qualsiasi Whisper HuggingFace: `openai/whisper-small`, `openai/whisper-medium`, `openai/whisper-large-v3`, etc.

## Out of Scope

- Riscrittura dello script MLX esistente (resta funzionante per Apple Silicon).
- UI/Gradio per inferenza.
- Multi-GPU / DeepSpeed (single L4).
