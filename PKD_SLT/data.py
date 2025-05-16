


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
    # set vocab to tokenizer
    tokenizer[trg_lang].set_vocab(trg_vocab)
    sequence_encoder = {
        trg_lang: trg_vocab.sentences_to_ids,
    }

    if train_data is not None:
        train_data.sequence_encoder = sequence_encoder

    dev_data = None
    if "dev" in datasets and dev_path is not None:
        dev_subset = cfg.get("sample_dev_subset", -1)
        if "random_dev_subset" in cfg:
            logger.warning(
                "`random_dev_subset` option is obsolete. "
                "Please use `sample_dev_subset` instead."
            )
            dev_subset = cfg.get("random_dev_subset", dev_subset)
        logger.info("Loading dev set...")
        dev_data = build_dataset(
            dataset_type="plain",
            path=dev_path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="dev",
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=dev_subset,
            **dataset_cfg,
        )
        logger.info("Loading dev set OK...")
        # test data
    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("Loading test set...")
        test_data = build_dataset(
            dataset_type="plain",
            path=test_path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="test",
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=-1,  # no subsampling for test
            **dataset_cfg,
        )
    logger.info("Loading test set OK")
    logger.info("Data loaded.")

    # Log statistics of data and vocabulary
    logger.info("Train dataset: %s", train_data)
    logger.info("Valid dataset: %s", dev_data)
    logger.info(" Test dataset: %s", test_data)
    if train_data:
        trg = "\n\t[TRG] " + " ".join(
            train_data.get_item(idx=0, lang=train_data.trg_lang, is_train=False)
        )
        logger.info("First training example:%s",  trg)

    logger.info("First 10 Trg tokens: %s", trg_vocab.log_vocab(10))

    logger.info("Number of unique Trg tokens (vocab_size): %d", len(trg_vocab))

    # train_iter = train_data.make_iter(
    #     batch_size=1,
    #     batch_type="token",
    #     seed=42,
    #     shuffle=True,
    #     num_workers=0,
    #     device=torch.device('cuda', 0),
    #     eos_index=2,
    #     pad_index=1,
    # )
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # batch = next(iter(train_iter))
    # print(batch.src)
    # print(batch.src.shape)
    # print(batch.src_mask)
    # print(batch.src_mask.shape)
    # print(batch.trg)
    # print(batch.trg_mask)
    # print(batch.trg_mask.shape)
    # print(batch.trg.shape)

    return trg_vocab, train_data, dev_data, test_data

    #--------------------------------------测试训练迭代------------------------------

    #--------------------------------------测试训练迭代------------------------------



