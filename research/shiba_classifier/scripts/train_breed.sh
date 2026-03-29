#!/bin/bash
#SBATCH --job-name=breed-train
#SBATCH --partition=A100.40gb
#SBATCH --reservation=1204
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/breed_train_%j.log
#SBATCH --error=logs/breed_train_%j.log

# ── 環境設定 ──
export CUDA_VISIBLE_DEVICES=MIG-9e900285-1b4a-52f0-b9f8-f02ed2133a5e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "GPU       : $CUDA_VISIBLE_DEVICES"
echo "Start     : $(date)"
echo "Project   : $PROJECT_DIR"
echo "=============================="

# ── 学習実行 ──
python3 src/models/breed_classifier/train_model.py \
    --epochs 30 \
    --batch-size 32 \
    --unfreeze 3

echo "=============================="
echo "End       : $(date)"
echo "=============================="
