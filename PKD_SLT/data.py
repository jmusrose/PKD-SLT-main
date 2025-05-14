


from typing import Dict, Optional, Tuple
# from PKD_SLT.datasets import BaseDataset
from PKD_SLT.helpers_for_ddp import get_logger
from PKD_SLT.tokenizers import build_tokenizer
from PKD_SLT.datasets import build_dataset
logger = get_logger(__name__)

def load_data(cfg:Dict, datasets: list=None):
    # -> Tuple[Vocabulary, Vocabulary, Optional[BaseDataset],
    # Optional[BaseDataset], Optional[BaseDataset]]:
    assert len(datasets) > 0, datasets

    src_cfg = cfg["src"]
    trg_cfg = cfg["trg"]
    src_lang = src_cfg["lang"]
    trg_lang = trg_cfg["lang"]
    train_path = cfg.get("train", None)
    dev_path = cfg.get("dev", None)
    test_path = cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError("Please specify at least one data source path.")
    dataset_cfg = cfg.get("dataset_cfg", {})
    # build tokenizer
    logger.info("Building tokenizer...")
    tokenizer = build_tokenizer(cfg)
    if "train" in datasets and train_path is not None:
        train_subset = cfg.get("sample_train_subset", -1)
        logger.info("Loading train set...")
        build_dataset(
            dataset_type="plain",
            path=train_path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="train",
            tokenizer=tokenizer,
            random_subset=train_subset,
            **dataset_cfg,

        )


