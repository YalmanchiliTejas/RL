#!/bin/bash
#SBATCH --job-name=lunar_rollout_compare
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

#SBATCH --array=0-3

set -euo pipefail

module load anaconda
conda activate rl

cd /home/tyalaman/RL/DQN_PER_Double/

ALGOS=("dqn" "ddqn" "dqn_per" "ddqn_per")
ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}

ENV_ID="LunarLander-v3"
SEED=0

MODEL_PATH="runs/${ALGO}_seed${SEED}_${ENV_ID}/q_network.pt"
VIDEO_DIR="rollouts/${ALGO}_seed${SEED}_${ENV_ID}"

echo "Running rollout for algo: $ALGO"
echo "Model path: $MODEL_PATH"
echo "Video dir: $VIDEO_DIR"
echo "SLURM job id: $SLURM_JOB_ID"
echo "SLURM array task id: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model file not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$VIDEO_DIR"

python rollout.py \
  --env "$ENV_ID" \
  --model "$MODEL_PATH" \
  --video-dir "$VIDEO_DIR" \
  --episodes 50 \
  --video-episodes 5 \
  --seed 100

echo "Finished rollout for algo: $ALGO"