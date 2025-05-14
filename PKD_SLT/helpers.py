import shutil
import functools
import operator
from pathlib import Path
import re
import unicodedata
from torch import Tensor, nn
from typing import List, Any
import numpy as np


def make_model_dir(model_dir: Path, overwrite: bool = False) -> None:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    """
    model_dir = model_dir.absolute()
    if model_dir.is_dir():
        if not overwrite:
            raise FileExistsError(
                f"Model directory {model_dir} exists "
                f"and overwriting is disabled."
            )
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)  # create model_dir recursively


def remove_extra_spaces(s: str) -> str:
    """
    Remove extra spaces
    - used in pre_process() / post_process() in tokenizer.py

    :param s: input string
    :return: string w/o extra white spaces
    """
    s = re.sub("\u200b", "", s)
    s = re.sub("[ 　]+", " ", s)

    s = s.replace(" ?", "?")
    s = s.replace(" !", "!")
    s = s.replace(" ,", ",")
    s = s.replace(" .", ".")
    s = s.replace(" :", ":")
    return s.strip()

def unicode_normalize(s: str) -> str:
    """
    apply unicodedata NFKC normalization
    - used in pre_process() in tokenizer.py

    :param s: input string
    :return: normalized string
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'")
    s = s.replace("“", '"')
    s = s.replace("”", '"')
    return s


def adjust_mask_size(mask: Tensor, batch_size: int, hyp_len: int) -> Tensor:
    """
    Adjust mask size along dim=1. used for forced decoding (trg prompting).

    :param mask: trg prompt mask in shape (batch_size, hyp_len)
    :param batch_size:
    :param hyp_len:
    """
    if mask is None:
        return None

    if mask.size(1) < hyp_len:
        _mask = mask.new_zeros((batch_size, hyp_len))
        _mask[:, :mask.size(1)] = mask
    elif mask.size(1) > hyp_len:
        _mask = mask[:, :hyp_len]
    else:
        _mask = mask
    assert _mask.size(1) == hyp_len, (_mask.size(), batch_size, hyp_len)
    return _mask


def read_list_from_file(input_path: Path) -> List[str]:
    """
    Read list of str from file in `input_path`.

    :param input_path: input file path
    :return: list of strings
    """
    if input_path is None:
        return []
    return [
        line.rstrip("\n")
        for line in input_path.read_text(encoding="utf-8").splitlines()
    ]


def flatten(array: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested 2D list. faster even with a very long array than
    [item for subarray in array for item in subarray] or newarray.extend().

    :param array: a nested list
    :return: flattened list
    """
    return functools.reduce(operator.iconcat, array, [])


def write_list_to_file(output_path: Path, array: List[Any]) -> None:
    """
    Write list of str to file in `output_path`.

    :param output_path: output file path
    :param array: list of strings
    """
    with output_path.open("w", encoding="utf-8") as opened_file:
        for entry in array:
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            opened_file.write(f"{entry}\n")