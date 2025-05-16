

from PKD_SLT.config import (
    BaseConfig
)
from PKD_SLT.data import load_data




def prepare (args: BaseConfig, rank:int, mode:str):
    # load the data
    if mode == "train":
        datasets = ["train", "dev", "test"]
    if mode == "test":
        datasets = ["dev", "test"]
    if mode == "translate":
        datasets = ["stream"]

    if mode != "train":
        if "voc_file" not in args.data["src"] or not args.data["src"]["voc_file"]:
            args.data["src"]["voc_file"] = (args.model_dir / "src_vocab.txt").as_posix()
        if "voc_file" not in args.data["trg"] or not args.data["trg"]["voc_file"]:
            args.data["trg"]["voc_file"] = (args.model_dir / "trg_vocab.txt").as_posix()
    trg_vocab, train_data, dev_data, test_data = load_data(args.data,datasets=datasets)

    if mode == "train" and rank == 0:
        trg_vocab.to_file(args.model_dir / "trg_vocab.txt")
        if hasattr(train_data.tokenizer[train_data.trg_lang], "copy_cfg_file"):
            train_data.tokenizer[train_data.trg_lang].copy_cfg_file(args.model_dir)

