import logging
import math
import os
from typing import Optional, Union, Dict

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler




def ddp_setup(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: int = 12355,
) -> None:
    """
    Setup distributed environment

    :param rank: Unique identifier of each process
    :param world_size: Total number of processes
    :param master_addr:
    :param master_port:
    """
    if dist.is_available():
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = master_addr
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

