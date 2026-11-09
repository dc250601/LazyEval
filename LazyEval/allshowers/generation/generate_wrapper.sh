#!/bin/bash
#SBATCH --job-name=ignorance-4-v100
#SBATCH --partition=maxgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=V100&GPUx4
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=LazyEval-%j.out
#SBATCH --error=LazyEval-%j.err


allocated_hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=${allocated_hosts%%$'\n'*}
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

exec srun --ntasks="$SLURM_NTASKS" --ntasks-per-node=4 \
    --gpu-bind=none --kill-on-bad-exit=1 \
    $HOME/AllShowers/bin/python -m LazyEval.allshowers.generation.generateshower "$@"