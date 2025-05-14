


from typing import Dict, Optional, Tuple
from torch.multiprocessing import cpu_count
import torch

# from PKD_SLT.datasets import BaseDataset
from PKD_SLT.helpers_for_ddp import get_logger
from PKD_SLT.tokenizers import build_tokenizer
from PKD_SLT.datasets import build_dataset
from PKD_SLT.vocabulary import build_vocab
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
        train_data = build_dataset(
            dataset_type="plain",
            path=train_path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="train",
            tokenizer=tokenizer,
            random_subset=train_subset,
            **dataset_cfg,
        )
    trg_vocab = build_vocab(cfg, dataset=train_data)

    sequence_encoder = {
        trg_lang: trg_vocab.sentences_to_ids,
    }

    if train_data is not None:
        train_data.sequence_encoder = sequence_encoder
    # set vocab to tokenizer
    tokenizer[trg_lang].set_vocab(trg_vocab)
    train_iter = train_data.make_iter(
        batch_size=1,
        batch_type="token",
        seed=42,
        shuffle=True,
        num_workers=0,
        device=torch.device('cuda', 0),
        eos_index=2,
        pad_index=1,
    )
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    batch = next(iter(train_iter))
    print(batch.src)
    print(batch.src.shape)
    print(batch.trg)
    print(batch.trg.shape)




