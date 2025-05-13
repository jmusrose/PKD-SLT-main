

from PKD_SLT.config import (
    BaseConfig
)





def prepare (arg: BaseConfig, rank:int, mode:str):
    # load the data
    if mode == "train":
        datasets = ["train", "dev", "test"]
    if mode == "test":
        datasets = ["dev", "test"]
    if mode == "translate":
        datasets = ["stream"]