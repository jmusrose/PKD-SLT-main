import shutil
from pathlib import Path
import re
import unicodedata


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