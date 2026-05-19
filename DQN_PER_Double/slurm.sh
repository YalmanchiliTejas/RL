#!/bin/bash
#SBATCH --job-name=lunar_dqn_compare
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

#SBATCH --array=0-3

set -euo pipefail

module load anaconda
conda activate rl

cd /home/tyalaman/RL/DQN_PER_Double/

ALGOS=("dqn" "ddqn" "dqn_per" "ddqn_per")
ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}

echo "Running algo: $ALGO"
echo "SLURM job id: $SLURM_JOB_ID"
echo "SLURM array task id: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python train.py \
  --env LunarLander-v3 \
  --algo "$ALGO"