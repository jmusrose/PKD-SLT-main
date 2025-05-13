


from typing import Dict, Optional, Tuple
from PKD_SLT.datasets import BaseDataset
from PKD_SLT.helpers_for_ddp import get_logger
from PKD_SLT.tokenizers import build_tokenizer
logger = get_logger(__name__)

def load_data(cfg:Dict, datasets: list=None):
    # -> Tuple[Vocabulary, Vocabulary, Optional[BaseDataset],
    # Optional[BaseDataset], Optional[BaseDataset]]:
    assert len(datasets) > 0, datasets

    trg_cfg = cfg["trg"]
    trg_lang = trg_cfg["lang"]
    train_path = cfg.get("train", None)
    dev_path = cfg.get("dev", None)
    test_path = cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError("Please specify at least one data source path.")

    # build tokenizer
    logger.info("Building tokenizer...")
    tokenizer = build_tokenizer(cfg)


