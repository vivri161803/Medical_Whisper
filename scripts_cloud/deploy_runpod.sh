#!/usr/bin/env bash
# =============================================================================
# deploy_runpod.sh — Setup & lancio fine-tuning Whisper su RunPod L4
# =============================================================================
#
# Uso:
#   1. Crea un pod RunPod con template PyTorch 2.4.0 + CUDA 12.1
#   2. SSH nel pod: ssh root@<pod-ip> -p <port>
#   3. Esegui questo script: bash deploy_runpod.sh
#
# Prerequisiti locali (da eseguire PRIMA di SSH):
#   rsync -avz --progress \
#     --include='scripts_cloud/***' \
#     --include='data/preprocessed/***' \
#     --include='data/medical_terms.txt' \
#     --exclude='*' \
#     ./ root@<pod-ip>:/workspace/whisper-finetune/
#
# Oppure usa scp per i singoli file:
#   scp -r scripts_cloud/ root@<pod-ip>:/workspace/whisper-finetune/
#   scp -r data/preprocessed/ root@<pod-ip>:/workspace/whisper-finetune/data/
#   scp data/medical_terms.txt root@<pod-ip>:/workspace/whisper-finetune/data/
#
# =============================================================================

set -euo pipefail

WORKSPACE="/workspace/whisper-finetune"
SCRIPTS_DIR="${WORKSPACE}/scripts_cloud"
VENV_DIR="${WORKSPACE}/.venv"

echo "============================================================"
echo "🚀 RunPod Setup — Whisper LoRA Fine-Tuning"
echo "============================================================"

# --- 1. Verifica GPU ---
echo ""
echo "🖥️  GPU disponibili:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || {
    echo "❌ nvidia-smi non trovato. Assicurati di essere su un pod GPU."
    exit 1
}

# --- 2. Crea workspace ---
echo ""
echo "📁 Workspace: ${WORKSPACE}"
mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"

# --- 3. Installa dipendenze ---
echo ""
echo "📦 Installazione dipendenze..."
pip install --upgrade pip
pip install -r "${SCRIPTS_DIR}/requirements.txt"

# --- 4. Verifica dati ---
echo ""
echo "📂 Verifica dati preprocessati..."
TRAIN_COUNT=$(find data/preprocessed/train -name "*.npz" 2>/dev/null | wc -l)
VAL_COUNT=$(find data/preprocessed/val -name "*.npz" 2>/dev/null | wc -l)

if [ "$TRAIN_COUNT" -eq 0 ]; then
    echo "❌ Nessun file .npz trovato in data/preprocessed/train/"
    echo "   Copia i dati preprocessati dal tuo Mac:"
    echo "   scp -r data/preprocessed/ root@<pod-ip>:${WORKSPACE}/data/"
    exit 1
fi

echo "   Training:   ${TRAIN_COUNT} samples"
echo "   Validation: ${VAL_COUNT} samples"

# --- 5. Verifica glossario ---
if [ -f "data/medical_terms.txt" ]; then
    TERMS_COUNT=$(grep -c -v '^#' data/medical_terms.txt | tr -d ' ')
    echo "   Glossario:  ${TERMS_COUNT} termini medici"
else
    echo "⚠️  Glossario medico non trovato (data/medical_terms.txt)"
fi

# --- 6. Login W&B ---
echo ""
echo "📊 Weights & Biases:"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "   ⚠️  WANDB_API_KEY non impostata."
    echo "   Per abilitare il logging, esegui:"
    echo "   export WANDB_API_KEY=<your-key>"
    echo "   Oppure: wandb login"
else
    echo "   ✅ WANDB_API_KEY trovata."
fi

# --- 7. Lancio training ---
echo ""
echo "============================================================"
echo "🎯 Lancio fine-tuning..."
echo "============================================================"
echo ""

cd "${SCRIPTS_DIR}"

python finetune_pytorch.py \
    --config training_config.yaml \
    "$@"

# --- 8. Risultati ---
echo ""
echo "============================================================"
echo "💾 Risultati salvati in: ${WORKSPACE}/outputs_cloud/"
echo ""
echo "Per scaricare i risultati sul tuo Mac:"
echo "  scp -r root@<pod-ip>:${WORKSPACE}/outputs_cloud/ ./outputs_cloud_results/"
echo "============================================================"
