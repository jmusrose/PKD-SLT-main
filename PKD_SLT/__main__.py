import argparse
import os
import shutil
import sys

import torch
import torch.multiprocessing as mp

from pathlib import Path
from PKD_SLT.helpers import make_model_dir
from PKD_SLT.config import load_config, _check_path
from PKD_SLT.helpers_for_ddp import get_logger
from PKD_SLT.training import train


def main():
    ap = argparse.ArgumentParser("PDK-SLT")
    ap.add_argument(
        "mode",
        choices=["train", "test", "translate"],
        help="Train a model or test or translate"
    )
    ap.add_argument(
        "config_path", metavar="config-path", type=str, help="Path to YAML config file"
    )

    ap.add_argument(
        "-o", "--output-path", type=str, help="Path for saving translation output"
    )

    ap.add_argument(
        "-a",
        "--save-attention",
        action="store_true",
        help="Save attention visualizations"
    )

    ap.add_argument("-s", "--save-scores", action="store_true", help="Save scores")

    ap.add_argument(
        "-t", "--skip-test", action="store_true", help="Skip test after training"
    )

    ap.add_argument(
        "-d", "--use-ddp", action="store_true", help="Invoke DDP environment"
    )
    args = ap.parse_args()
    cfg = load_config(Path(args.config_path))

    # 创建模型保存目录

    if args.mode == "train":
        make_model_dir(
            Path(cfg["model_dir"]), overwrite=cfg["training"].get("overwrite", False))
    model_dir = _check_path(cfg["model_dir"], allow_empty=False)
    # 复制配置文件到模型目录
    if args.mode == "train":
        shutil.copy2(args.config_path, (model_dir / "config.yaml").as_posix())
    # 编写训练日志
    logger = get_logger("",log_file=Path(model_dir / f"{args.mode}.log").as_posix())
    logger.info("您好，欢迎使用PKD-SLT")

    # 多卡并行
    if args.use_ddp:
        n_gpu = torch.cuda.device_count() \
            if cfg.get("use_cuda", False) and torch.cuda.is_available() else 0
        if args.mode == "train":
            assert n_gpu > 1, "For DDP training, `world_size` must be > 1."
            logger.info("Spawn torch.multiprocessing (nprocs=%d).", n_gpu)
            cfg["use_ddp"] = args.use_ddp
            mp.spawn(train, args=(n_gpu, cfg, args.skip_test), nprocs=n_gpu)
        elif args.mode == "test":
            raise RuntimeError("For testing mode, DDP is currently not available.")
        elif args.mode == "translate":
            raise RuntimeError(
                "For interactive translation mode, "
                "DDP is currently not available."
            )
    else:
        if args.mode == "train":
            # train(rank=0, world_size=None, cfg=cfg)
            train(rank=0, world_size=None, cfg=cfg, skip_test=args.skip_test)


if __name__ == "__main__":
    main()
