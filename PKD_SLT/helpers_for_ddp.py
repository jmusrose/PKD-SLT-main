import logging
import math
import os
from typing import Optional, Union

import torch

import torch.distributed as dist

def use_ddp() -> bool:
    """Check if DDP environment is available"""
    return dist.is_available() and dist.is_initialized()

class MultiProcessAdapter(logging.LoggerAdapter):
    """
    An adapter to assist with logging in multiprocess.

    taken from Huggingface's Accelerate logger
    """

    def log(self, level, msg, *args, **kwargs):
        """
        Delegates logger call after checking if we should log.
        """
        flag = False
        master_only = kwargs.pop("master_only", True)

        if master_only:
            rank = dist.get_rank() if use_ddp() else 0
            flag = rank == 0

        if self.isEnabledFor(level) and flag:
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str = "", log_file: str = None) -> logging.Logger:
    """
    Create a logger for logging the training/testing process.

    :param name: logger name.
    :param log_file: path to file where log is stored as well
    :return: logging.Logger
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    def _add_filehandler(logger, log_file):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def _add_streamhandler(logger):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    # assign file handler whenever `log_file` arg is provided
    if log_file is not None:
        for logger_name in logging.root.manager.loggerDict:
            if logger_name.startswith("joeynmt."):
                logger = logging.getLogger(logger_name)
                if len(logger.handlers) < 2:
                    _add_filehandler(logger, log_file)

    current_logger = logging.getLogger(name)
    if len(current_logger.handlers) == 0:
        current_logger.setLevel(level=logging.DEBUG)
        _add_streamhandler(current_logger)
        if log_file is not None:
            _add_filehandler(current_logger, log_file)

    current_logger.propagate = False  # otherwise root logger prints things again

    return MultiProcessAdapter(current_logger, {})