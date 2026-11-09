#!/bin/bash
#SBATCH --job-name=Generation-4-v100
#SBATCH --partition=maxgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=V100&GPUx4
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=LazyEval-%j.out
#SBATCH --error=LazyEval-%j.err


allocated_hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=${allocated_hosts%%$'\n'*}
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

exec srun \
    --export=ALL \
    --ntasks="$SLURM_NTASKS" \
    --gpu-bind=none \
    --kill-on-bad-exit=1 \
    /usr/bin/env \
        MASTER_ADDR="$MASTER_ADDR" \
        MASTER_PORT="$MASTER_PORT" \
        "$HOME/AllShowers/bin/python" \
        -m LazyEval.allshowers.generation.generateshower "$@"