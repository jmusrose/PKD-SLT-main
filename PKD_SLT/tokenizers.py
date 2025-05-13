

from typing import Dict
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