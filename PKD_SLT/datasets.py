from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import os
import torch
import pickle
import numpy as np
import gzip
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from PKD_SLT.tokenizers import BasicTokenizer
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from PKD_SLT.config import ConfigurationError
from PKD_SLT.batch import Batch
from PKD_SLT.helpers_for_ddp import get_logger
from PKD_SLT.helpers_for_ddp import (
    DistributedSubsetSampler,
    RandomSubsetSampler,
    get_logger,
    use_ddp,
)

logger = get_logger(__name__)
CPU_DEVICE = torch.device("cpu")


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

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
        # assert self.sequence_encoder[self.src_lang] is not None
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
            f'trg_lang="{self.trg_lang}", '
            f" random_subset={self.random_subset}, "
        )

class SignDataset(BaseDataset):
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
        max_seq_length: Optional[int] = None,
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
        self.max_seq_length = max_seq_length
        # load data
        self.data = self.load_data(path, **kwargs)
        self.reset_indices()


    def load_data(self, path: str, **kwargs) -> Any:

        data = load_dataset_file(path)
        sign_data = {}

        # Ensure we have 'sign' data as src and 'translation' as trg
        sign_data[self.src_lang] = []
        if self.has_trg:
            sign_data[self.trg_lang] = []
            # Process each item in the data
        del data[0]
        for item in data:
            # Convert sign tensor from list to numpy array
            if isinstance(item['sign'], list):
                # If sign is a list of tensors, we take the first one
                sign_tensor = np.array(item['sign'][0], dtype=np.float32)
            else:
                sign_tensor = np.array(item['sign'], dtype=np.float32)

            # Apply sequence length constraints if specified
            if self.max_seq_length is not None:
                if sign_tensor.shape[0] > self.max_seq_length:
                    # Truncate if too long
                    sign_tensor = sign_tensor[:self.max_seq_length, :]
                elif sign_tensor.shape[0] < self.max_seq_length:
                    # Pad with zeros if too short
                    padding = np.zeros(
                        (self.max_seq_length - sign_tensor.shape[0], sign_tensor.shape[1]),
                        dtype=np.float32
                    )
                    sign_tensor = np.concatenate([sign_tensor, padding], axis=0)

            sign_data[self.src_lang].append(sign_tensor)

            if self.has_trg:
                translation = item.get('translation', '')
                # Apply text preprocessing if needed
                if self.tokenizer[self.trg_lang] is not None:
                    translation = self.tokenizer[self.trg_lang].pre_process(translation)
                sign_data[self.trg_lang].append(translation)

        return sign_data


    def lookup_item(self, idx: int, lang: str) -> Tuple[Any, str]:
        """
        Look up one item of the given index and language.

        :param idx: index of the item to look up
        :param lang: language code
        :return: (item, prompt) tuple where item is the sign tensor or text translation
        """
        try:
            item = self.data[lang][idx]
            # No prompt for sign language data
            prompt = None
            return item, prompt
        except Exception as e:
            logger.error(f"Error looking up item {idx} for language {lang}: {e}")
            raise ValueError from e


    def get_item(self, idx: int, lang: str, is_train: bool = None) -> Any:
        """
        Get one src/trg item of the given index.

        :param idx: index of the item to get
        :param lang: language code
        :param is_train: boolean indicating if it's training mode
        :return: sign tensor or tokenized text
        """
        item, prompt = self.lookup_item(idx, lang)

        # For target language, apply tokenization
        if lang == self.trg_lang and self.tokenizer[lang] is not None:
            is_train = self.split == "train" if is_train is None else is_train
            item = self.tokenizer[lang](item, is_train=is_train)

        # For source language (sign), item is already a tensor
        return item

    def get_list(self, lang: str, tokenized: bool = False, subsampled: bool = True) -> Union[List[Any], List[str]]:
        """
        Return list of items in the given language.

        :param lang: language code
        :param tokenized: whether to tokenize the text (only applicable for trg)
        :param subsampled: whether to apply subsampling
        :return: list of sign tensors or text translations
        """
        indices = self.indices if subsampled else range(self.__len__())
        item_list = []

        for idx in indices:
            item, _ = self.lookup_item(idx, lang)

            if lang == self.trg_lang and tokenized and self.tokenizer[lang] is not None:
                item = self.tokenizer[lang](item, is_train=False)

            item_list.append(item)

        assert len(indices) == len(item_list), (len(indices), len(item_list))
        return item_list

    def collate_fn(
        self,
        batch: List[Tuple],
        pad_index: int,
        eos_index: int,
        device: torch.device = CPU_DEVICE,
    ) -> Batch:
        """
        Custom collate function for sign language data.

        :param batch: list of (idx, src, trg) tuples
        :param pad_index: padding token index
        :param eos_index: end of sequence token index
        :param device: target device
        :return: Batch object
        """
        idx, src_list, trg_list = zip(*batch)
        assert len(batch) == len(src_list) == len(trg_list), (len(batch), len(src_list))
        assert all(s is not None for s in src_list), src_list

        # handle source (sign features)
        max_src_len = max(x.shape[0] for x in src_list)
        feature_dim = src_list[0].shape[1]
        src_lengths = []
        padded_src = []

        for sign_tensor in src_list:
            src_len = sign_tensor.shape[0]
            src_lengths.append(src_len)

            if src_len < max_src_len:
                padding = np.zeros((max_src_len - src_len, feature_dim), dtype=np.float32)
                padded_tensor = np.concatenate([sign_tensor, padding], axis=0)
            else:
                padded_tensor = sign_tensor

            padded_src.append(padded_tensor)

        src = torch.tensor(np.array(padded_src)).float()
        src_length = torch.tensor(src_lengths).long()

        # handle target (text translations)
        if self.has_trg or self.has_prompt[self.trg_lang]:
            if self.has_trg:
                assert all(t is not None for t in trg_list), trg_list
            trg, _, trg_prompt_mask = self.sequence_encoder[self.trg_lang](
                trg_list, bos=True, eos=self.has_trg
            )
        else:
            assert all(t is None for t in trg_list)
            trg, trg_prompt_mask = None, None

        return Batch(
            src=src,
            src_length=src_length,
            src_prompt_mask=None,  # no source prompt mask for sign language
            trg=torch.tensor(trg).long() if trg is not None else None,
            trg_prompt_mask=torch.tensor(trg_prompt_mask).long() if trg_prompt_mask is not None else None,
            indices=torch.tensor(idx).long(),
            device=device,
            pad_index=pad_index,
            eos_index=eos_index,
            is_train=self.split == "train",
        )


    def __len__(self) -> int:
        """
        Return the size of the dataset.

        :return: Number of samples in the dataset
        """
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
    dataset = SignDataset(
        path=path,
        src_lang=src_lang,
        trg_lang=trg_lang,
        split=split,
        has_trg=True,
        tokenizer=tokenizer,
        sequence_encoder=sequence_encoder,
        random_subset=random_subset,
        **kwargs,
    )
    return dataset


class SentenceBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on num of instances.
    An instance longer than dataset.max_len will be filtered out.

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If `True`, the sampler will drop the last batch if its size
        would be less than `batch_size`
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool, seed: int):
        super().__init__(sampler, batch_size, drop_last)
        self.seed = seed

    @property
    def num_samples(self) -> int:
        """
        Returns number of samples in the dataset.
        This may change during sampling.

        Note: len(dataset) won't change during sampling.
              Use len(dataset) instead, to retrieve the original dataset length.
        """
        assert self.sampler.data_source.indices is not None
        try:
            return len(self.sampler)
        except NotImplementedError as e:  # pylint: disable=unused-variable # noqa: F841
            return len(self.sampler.data_source.indices)

    def __iter__(self):
        batch = []
        d = self.sampler.data_source

        for idx in self.sampler:
            _, src, trg = d[idx]  # pylint: disable=unused-variable
            if src is not None:  # otherwise drop instance
                batch.append(idx)

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

        if len(batch) > 0:
            if not self.drop_last:
                yield batch
            else:
                logger.warning(f"Drop indices {batch}.")

    def __len__(self) -> int:
        # pylint: disable=no-else-return
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def set_seed(self, seed: int) -> None:
        assert seed is not None, seed
        self.sampler.data_source.seed = seed

        if hasattr(self.sampler, 'set_seed'):
            self.sampler.set_seed(seed)  # set seed and resample
        elif hasattr(self.sampler, 'generator'):
            self.sampler.generator.manual_seed(seed)

        if self.num_samples < len(self.sampler.data_source):
            logger.info(
                "Sample random subset from %s data: n=%d, seed=%d",
                self.sampler.data_source.split, self.num_samples, seed
            )

    def reset(self) -> None:
        if hasattr(self.sampler, 'reset'):
            self.sampler.reset()

    def get_state(self):
        if hasattr(self.sampler, 'generator'):
            return self.sampler.generator.get_state()
        return None

    def set_state(self, state) -> None:
        if hasattr(self.sampler, 'generator'):
            self.sampler.generator.set_state(state)


class TokenBatchSampler(SentenceBatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on num of tokens
    (incl. padding). An instance longer than dataset.max_len or shorter than
    dataset.min_len will be filtered out.
    * no bucketing implemented

    .. warning::
        In DDP, we shouldn't use TokenBatchSampler for prediction, because we cannot
        ensure that the data points will be distributed evenly across devices.
        `ddp_merge()` (`dist.all_gather()`) called in `predict()` can get stuck.

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If `True`, the sampler will drop the last batch if
            its size would be less than `batch_size`
    """

    def __iter__(self):
        """yields list of indices"""
        batch = []
        max_tokens = 0
        d = self.sampler.data_source

        for idx in self.sampler:
            _, src, trg = d[idx]  # call __getitem__()
            if src is not None:  # otherwise drop instance
                src_len = 0 if src is None else len(src)
                trg_len = 0 if trg is None else len(trg)
                n_tokens = 0 if src_len == 0 else max(src_len + 1, trg_len + 1)
                batch.append(idx)

                if n_tokens > max_tokens:
                    max_tokens = n_tokens
                if max_tokens * len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                    max_tokens = 0

        if len(batch) > 0:
            if not self.drop_last:
                yield batch
            else:
                logger.warning(f"Drop indices {batch}.")

    def __len__(self):
        raise NotImplementedError