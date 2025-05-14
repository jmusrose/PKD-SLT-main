

from typing import Dict, Union, List
from PKD_SLT.helpers_for_ddp import get_logger
from PKD_SLT.helpers import remove_extra_spaces, unicode_normalize
logger = get_logger(__name__)


class BasicTokenizer:
    SPACE = chr(32)  # ' ': half-width white space (ascii)
    SPACE_ESCAPE = chr(9601)  # '▁': sentencepiece default

    def __init__(
            self,
            level: str = "word",
            lowercase: bool = False,
            normalize: bool = False,
            max_length: int = -1,
            min_length: int = -1,
            **kwargs,
    ):
        # pylint: disable=unused-argument
        self.level = level
        self.lowercase = lowercase
        self.normalize = normalize

        # filter by length
        self.max_length = max_length
        self.min_length = min_length

    def pre_process(self, raw_input: str, allow_empty: bool = False) -> str:
        if not allow_empty:
            assert isinstance(raw_input, str) and raw_input.strip() != "", \
                "The input sentence is empty! Please make sure " \
                "that you are feeding a valid input."
        if self.normalize:
            raw_input = remove_extra_spaces(unicode_normalize(raw_input))
        if self.lowercase:
            raw_input = raw_input.lower()

        if not allow_empty:
            # ensure the string is not empty.
            assert raw_input is not None and len(raw_input) > 0, raw_input

        return raw_input

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize single sentence"""
        if raw_input is None:
            return None

        if self.level == "word":
            sequence = raw_input.split(self.SPACE)

        if is_train and self._filter_by_length(len(sequence)):
            return None
        return sequence

    def _filter_by_length(self, length: int) -> bool:
        """
        Check if the given seq length is out of the valid range.

        :param length: (int) number of tokens
        :return: True if the length is invalid(= to be filtered out), False if valid.
        """
        return length > self.max_length > 0 or self.min_length > length > 0

    def _remove_special(self, sequence: List[str], generate_unk: bool = False):
        specials = self.specials if generate_unk else self.specials + [self.unk_token]
        valid = [token for token in sequence if token not in specials]
        if len(valid) == 0:  # if empty, return <unk>
            valid = [self.unk_token]
        return valid

    def post_process(
            self,
            sequence: Union[List[str], str],
            generate_unk: bool = True,
            cut_at_sep: bool = True
        ) -> str:
        if isinstance(sequence, list):
            if cut_at_sep:
                try:
                    sep_pos = sequence.index(self.sep_token)  # cut off prompt
                    sequence = sequence[sep_pos + 1:]

                except ValueError as e:  # pylint: disable=unused-variable # noqa: F841
                    pass
            sequence = self._remove_special(sequence, generate_unk=generate_unk)
            sequence = self.SPACE.join(sequence)
            # Remove extra spaces
        if self.normalize:
            sequence = remove_extra_spaces(sequence)

        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence

    def set_vocab(self, vocab) -> None:
        """
        Set vocab
        :param vocab: (Vocabulary)
        """
        # pylint: disable=attribute-defined-outside-init
        self.unk_token = vocab.specials[vocab.unk_index]
        self.eos_token = vocab.specials[vocab.eos_index]
        self.sep_token = vocab.specials[vocab.sep_index] if vocab.sep_index else None
        specials = vocab.specials + vocab.lang_tags
        self.specials = [token for token in specials if token != self.unk_token]
        self.lang_tags = vocab.lang_tags

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(level={self.level}, "
            f"lowercase={self.lowercase}, normalize={self.normalize}, "
            f"filter_by_length=({self.min_length}, {self.max_length}), "
            # f"pretokenizer={self.pretokenizer})"
        )


def _build_tokenizer(cfg: Dict) -> BasicTokenizer:
    tokenizer = None
    tokenizer_cfg = cfg.get("tokenizer_cfg", {})
    tokenizer = BasicTokenizer(
        level=cfg["level"],
        lowercase=cfg.get("lowercase", False),
        normalize=cfg.get("normalize", False),
        max_length=cfg.get("max_length", -1),
        min_length=cfg.get("min_length", -1),
        **tokenizer_cfg,
    )
    return tokenizer


def build_tokenizer(cfg: Dict) -> Dict[str, BasicTokenizer]:
    # src_lang = cfg["src"]["lang"]
    trg_lang = cfg["trg"]["lang"]
    tokenizer = {
        # src_lang: _build_tokenizer(cfg["src"]),
        trg_lang: _build_tokenizer(cfg["trg"]),
    }
    # logger.info("%s tokenizer: %s", src_lang, tokenizer[src_lang])
    logger.info("%s tokenizer: %s", trg_lang, tokenizer[trg_lang])
    return tokenizer