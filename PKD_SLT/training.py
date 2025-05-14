from typing import Dict
from pathlib import Path

from PKD_SLT.helpers_for_ddp import get_logger
from PKD_SLT.version import ddp_setup
from PKD_SLT.config import log_config, parse_global_args
from PKD_SLT.prediction import prepare

def train(rank: int, world_size: int, cfg: Dict, skip_test: bool = False):



    if cfg.pop("use_ddp", False):
        # initialize ddp
        # TODO: make `master_addr` and `master_port` configurable
        ddp_setup(rank, world_size, master_addr="localhost", master_port=12355)

        # need to assign file handlers again, after multi-processes are spawned...
        get_logger(__name__, log_file=Path(cfg["model_dir"]) / "train.log")

    # 编写日志
    log_config(cfg)
    args = parse_global_args(cfg, rank=rank, mode="train")
    # print(args)
    prepare(args, rank=rank, mode="train")


