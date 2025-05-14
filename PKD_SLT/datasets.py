from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from PKD_SLT.tokenizers import BasicTokenizer





class BaseDataset(Dataset):
    """
    BaseDataset which loads and looks up data.
    - holds pointer to tokenizers, encoding functions.

    :param path: path to data directory
    :param src_lang: source language code, i.e. `en`
    :param trg_lang: target language code, i.e. `de`
    :param has_trg: bool indicator if trg exists
    :param has_prompt: bool indicator if prompt exists
    :param split: bool indicator for train set or not
    :param tokenizer: tokenizer objects
    :param sequence_encoder: encoding functions
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: str = "train",
        has_trg: bool = False,
        has_prompt: Dict[str, bool] = None,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1
    ):

        self.path = path
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.has_trg = has_trg
        self.split = split
        if self.split == "train":
            assert self.has_trg

        self.tokenizer = tokenizer
        self.sequence_encoder = sequence_encoder
        self.has_prompt = has_prompt

        # for random subsampling
        self.random_subset = random_subset
        self.indices = None  # range(self.__len__())
        # Note: self.indices is kept sorted, even if shuffle = True in make_iter()
        # (Sampler yields permuted indices)
        self.seed = 1  # random seed for generator

    def reset_indices(self, random_subset: int = None):
        # should be called after data are loaded.
        # otherwise self.__len__() is undefined.
        self.indices = list(range(self.__len__())) if self.__len__() > 0 else []
        if random_subset is not None:
            self.random_subset = random_subset

        if 0 < self.random_subset:
            assert (self.split != "test" and self.random_subset < self.__len__()), \
                ("Can only subsample from train or dev set "
                 f"larger than {self.random_subset}.")

    def load_data(self, path: Path, **kwargs) -> Any:
        """
        load data
            - preprocessing (lowercasing etc) is applied here.
        """
        raise NotImplementedError

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        """
        seek one src/trg item of the given index.
            - tokenization is applied here.
            - length-filtering, bpe-dropout etc also triggered if self.split == "train"
        """

        # workaround if tokenizer prepends an extra escape symbol before lang_tang ...
        def _remove_escape(item):
            if (
                item  is not None and self.tokenizer[lang] is not None
                and item[0] == self.tokenizer[lang].SPACE_ESCAPE
                and item[1] in self.tokenizer[lang].lang_tags
            ):
                return item[1:]
            return item

        line, prompt = self.lookup_item(idx, lang)
        is_train = self.split == "train" if is_train is None else is_train
        item = _remove_escape(self.tokenizer[lang](line, is_train=is_train))

        if self.has_prompt[lang] and prompt is not None:
            prompt = _remove_escape(self.tokenizer[lang](prompt, is_train=False))
            item = item if item is not None else []

            max_length = self.tokenizer[lang].max_length
            if 0 < max_length < len(prompt) + len(item) + 1:
                # truncate prompt
                offset = max_length - len(item) - 1
                if prompt[0] in self.tokenizer[lang].lang_tags:
                    prompt = [prompt[0]] + prompt[-(offset - 1):]
                else:
                    prompt = prompt[-offset:]

            item = prompt + [self.tokenizer[lang].sep_token] + item
        return item

    def lookup_item(self, idx: int, lang: str) -> Tuple[str, str]:
        raise NotImplementedError

    def __getitem__(self, idx: Union[int, str]) -> Tuple[int, List[str], List[str]]:
        """
        lookup one item pair of the given index.

        :param idx: index of the instance to lookup
        :return:
            - index  # needed to recover the original order
            - tokenized src sentences
            - tokenized trg sentences
        """
        if idx > self.__len__():
            raise KeyError

        src, trg = None, None
        src = self.get_item(idx=idx, lang=self.src_lang)
        if self.has_trg or self.has_prompt[self.trg_lang]:
            trg = self.get_item(idx=idx, lang=self.trg_lang)
            if trg is None:
                src = None
        return idx, src, trg

    def get_list(self,
                 lang: str,
                 tokenized: bool = False,
                 subsampled: bool = True) -> Union[List[str], List[List[str]]]:
        """get data column-wise."""
        raise NotImplementedError

    @property
    def src(self) -> List[str]:
        """get detokenized preprocessed data in src language."""
        return self.get_list(self.src_lang, tokenized=False, subsampled=True)

    @property
    def trg(self) -> List[str]:
        """get detokenized preprocessed data in trg language."""
        return (
            self.get_list(self.trg_lang, tokenized=False, subsampled=True)
            if self.has_trg else []
        )

    def collate_fn(
        self,
        batch: List[Tuple],
        pad_index: int,
        eos_index: int,
        device: torch.device = CPU_DEVICE,
    ) -> Batch:
        """
        Custom collate function.
        See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn for details.
        Please override the batch class here. (not in TrainManager)

        :param batch:
        :param pad_index:
        :param eos_index:
        :param device:
        :return: joeynmt batch object
        """
        idx, src_list, trg_list = zip(*batch)
        assert len(batch) == len(src_list) == len(trg_list), (len(batch), len(src_list))
        assert all(s is not None for s in src_list), src_list
        src, src_length, src_prompt_mask = self.sequence_encoder[
            self.src_lang](src_list, bos=False, eos=True)

        if self.has_trg or self.has_prompt[self.trg_lang]:
            if self.has_trg:
                assert all(t is not None for t in trg_list), trg_list
            trg, _, trg_prompt_mask = self.sequence_encoder[self.trg_lang](
                trg_list, bos=True, eos=self.has_trg
            )  # no EOS if not self.has_trg
        else:
            assert all(t is None for t in trg_list)
            trg, trg_prompt_mask = None, None  # Note: we don't need trg_length!

        return Batch(
            src=torch.tensor(src).long(),
            src_length=torch.tensor(src_length).long(),
            src_prompt_mask=(
                torch.tensor(src_prompt_mask).long()
                if self.has_prompt[self.src_lang] else None
            ),
            trg=torch.tensor(trg).long() if trg else None,
            trg_prompt_mask=(
                torch.tensor(trg_prompt_mask).long()
                if self.has_prompt[self.trg_lang] else None
            ),
            indices=torch.tensor(idx).long(),
            device=device,
            pad_index=pad_index,
            eos_index=eos_index,
            is_train=self.split == "train",
        )

    def make_iter(
        self,
        batch_size: int,
        batch_type: str = "sentence",
        seed: int = 42,
        shuffle: bool = False,
        num_workers: int = 0,
        pad_index: int = 1,
        eos_index: int = 3,
        device: torch.device = CPU_DEVICE,
        generator_state: torch.Tensor = None,
    ) -> DataLoader:
        """
        Returns a torch DataLoader for a torch Dataset. (no bucketing)

        :param batch_size: size of the batches the iterator prepares
        :param batch_type: measure batch size by sentence count or by token count
        :param seed: random seed for shuffling
        :param shuffle: whether to shuffle the order of sequences before each epoch
                        (for testing, no effect even if set to True; generator is
                        still used for random subsampling, but not for permutation!)
        :param num_workers: number of cpus for multiprocessing
        :param pad_index:
        :param eos_index:
        :param device:
        :param generator_state:
        :return: torch DataLoader
        """
        shuffle = shuffle and self.split == "train"

        # for decoding in DDP, we cannot use TokenBatchSampler
        if use_ddp() and self.split != "train":
            assert batch_type == "sentence", self

        generator = torch.Generator()
        generator.manual_seed(seed)
        if generator_state is not None:
            generator.set_state(generator_state)

        # define sampler which yields an integer
        sampler: Sampler[int]
        if use_ddp():  # use ddp
            sampler = DistributedSubsetSampler(
                self, shuffle=shuffle, drop_last=True, generator=generator
            )
        else:
            sampler = RandomSubsetSampler(self, shuffle=shuffle, generator=generator)

        # batch sampler which yields a list of integers
        if batch_type == "sentence":
            batch_sampler = SentenceBatchSampler(
                sampler, batch_size=batch_size, drop_last=False, seed=seed
            )
        elif batch_type == "token":
            batch_sampler = TokenBatchSampler(
                sampler, batch_size=batch_size, drop_last=False, seed=seed
            )
        else:
            raise ConfigurationError(f"{batch_type}: Unknown batch type")

        # initialize generator seed
        batch_sampler.set_seed(seed)  # set seed and resample

        # ensure that sequence_encoder (padding func) exists
        assert self.sequence_encoder[self.src_lang] is not None
        if self.has_trg:
            assert self.sequence_encoder[self.trg_lang] is not None

        # data iterator
        return DataLoader(
            dataset=self,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                self.collate_fn,
                eos_index=eos_index,
                pad_index=pad_index,
                device=device
            ),
            num_workers=num_workers
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(split={self.split}, len={self.__len__()}, "
            f'src_lang="{self.src_lang}", trg_lang="{self.trg_lang}", '
            f"has_trg={self.has_trg}, random_subset={self.random_subset}, "
            f"has_src_prompt={self.has_prompt[self.src_lang]}, "
            f"has_trg_prompt={self.has_prompt[self.trg_lang]})"
        )

class PlaintextDataset(BaseDataset):
    """
    PlaintextDataset which stores plain text pairs.
    - used for text file data in the format of one sentence per line.
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: str = "train",
        has_trg: bool = False,
        has_prompt: Dict[str, bool] = None,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        **kwargs
    ):

        super().__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            has_prompt=has_prompt,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset
        )

        # load data
        self.data = self.load_data(path, **kwargs)
        self.reset_indices()

    def load_data(self, path: str, **kwargs) -> Any:

        def _pre_process(seq, lang):
            if self.tokenizer[lang] is not None:
                seq = [self.tokenizer[lang].pre_process(s) for s in seq if len(s) > 0]
            return seq

        path = Path(path)
        src_file = path.with_suffix(f"{path.suffix}.{self.src_lang}")
        assert src_file.is_file(), f"{src_file} not found. Abort."

        src_list = read_list_from_file(src_file)
        data = {self.src_lang: _pre_process(src_list, self.src_lang)}

        if self.has_trg:
            trg_file = path.with_suffix(f"{path.suffix}.{self.trg_lang}")
            assert trg_file.is_file(), f"{trg_file} not found. Abort."

            trg_list = read_list_from_file(trg_file)
            data[self.trg_lang] = _pre_process(trg_list, self.trg_lang)
            assert len(src_list) == len(trg_list)
        return data

    def lookup_item(self, idx: int, lang: str) -> Tuple[str, str]:
        try:
            line = self.data[lang][idx]
            prompt = (
                self.data[f"{lang}_prompt"][idx]
                if f"{lang}_prompt" in self.data else None
            )
            return line, prompt
        except Exception as e:
            logger.error(idx, e)
            raise ValueError from e

    def get_list(self,
                 lang: str,
                 tokenized: bool = False,
                 subsampled: bool = True) -> Union[List[str], List[List[str]]]:
        """
        Return list of preprocessed sentences in the given language.
        (not length-filtered, no bpe-dropout)
        """
        indices = self.indices if subsampled else range(self.__len__())
        item_list = []
        for idx in indices:
            item, _ = self.lookup_item(idx, lang)
            if tokenized:
                item = self.tokenizer[lang](item, is_train=False)
            item_list.append(item)
        assert len(indices) == len(item_list), (len(indices), len(item_list))
        return item_list

    def __len__(self) -> int:
        return len(self.data[self.src_lang])


def build_dataset(
            dataset_type: str,
            path: str,
            src_lang: str,
            trg_lang: str,
            split: str,
            tokenizer: Dict = None,
            sequence_encoder: Dict = None,
            random_subset: int = -1,
            **kwargs,
    ):
    dataset = None
    has_trg = True  # by default, we expect src-trg pairs
    _placeholder = {src_lang: None, trg_lang: None}
    tokenizer = _placeholder if tokenizer is None else tokenizer
    sequence_encoder = _placeholder if sequence_encoder is None else sequence_encoder
    if not Path(path).with_suffix(f"{Path(path).suffix}.{trg_lang}").is_file():
        has_trg = False  # no target is given -> create dataset from src only
        print("666")
        dataset = SignDataset(

        )


