import os
import socket

import numpy as np
import torch
import torch.distributed as dist

def setup_distributed():
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl",rank=rank,world_size=world_size)
    
    print(f"[rank {rank:03d}/{world_size}] "
        f"host={socket.gethostname()} "
        f"GPU={local_rank}",
        flush=True)

    return rank, local_rank, world_size, device
