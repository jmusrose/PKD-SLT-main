import torch
from torch import Tensor
from typing import Optional, List
import numpy as np


class Batch:
    """Object for holding a batch of sign language translation data."""

    def __init__(
            self,
            src: Tensor,  # sign features [batch_size, max_src_len, feature_dim]
            src_length: Tensor,  # source lengths [batch_size]
            src_prompt_mask: Optional[Tensor],
            trg: Optional[Tensor],  # target text [batch_size, max_trg_len]
            trg_prompt_mask: Optional[Tensor],
            indices: Tensor,
            device: torch.device,
            pad_index: int,
            eos_index: int,
            is_train: bool = True,
    ):
        self.src = src
        self.src_length = src_length
        # 为手语特征计算mask: [batch_size, 1, max_src_len]
        self.src_mask = (src_length.unsqueeze(1) > torch.arange(src.size(1)).to(src.device)).unsqueeze(1)
        self.src_prompt_mask = src_prompt_mask
        self.trg = None
        self.trg_mask = None
        self.trg_prompt_mask = None
        self.indices = indices

        self.nseqs = src.size(0)
        self.ntokens = None  # 用于normalization
        self.has_trg = trg is not None
        self.is_train = is_train

        if self.has_trg:
            self.trg = trg
            # 为文本计算mask: [batch_size, 1, max_trg_len]
            self.trg_mask = (self.trg != pad_index).unsqueeze(1)
            self.ntokens = self.trg_mask.sum().item()

            if trg_prompt_mask is not None:
                self.trg_prompt_mask = trg_prompt_mask

        if device.type == "cuda":
            self._make_cuda(device)

    def _make_cuda(self, device: torch.device) -> None:
        """Move the batch to GPU"""
        self.src = self.src.to(device)
        self.src_length = self.src_length.to(device)
        self.src_mask = self.src_mask.to(device)
        self.indices = self.indices.to(device)

        if self.src_prompt_mask is not None:
            self.src_prompt_mask = self.src_prompt_mask.to(device)

        if self.has_trg:
            self.trg = self.trg.to(device)
            self.trg_mask = self.trg_mask.to(device)
            if self.trg_prompt_mask is not None:
                self.trg_prompt_mask = self.trg_prompt_mask.to(device)

    def normalize(
            self,
            tensor: Tensor,
            normalization: str = "none",
            n_gpu: int = 1,
            n_accumulation: int = 1,
    ) -> Tensor:
        """
        Normalizes batch tensor (i.e. loss).

        :param tensor: tensor to normalize, i.e. batch loss
        :param normalization: one of {`batch`, `tokens`, `none`}
        :param n_gpu: number of gpus
        :param n_accumulation: number of gradient accumulation steps
        :return: normalized tensor
        """
        if tensor is None:
            return None
        assert torch.is_tensor(tensor)

        if n_gpu > 1:
            tensor = tensor.sum()

        if normalization == "sum":
            return tensor
        elif normalization == "batch":
            normalizer = self.nseqs
        elif normalization == "tokens":
            normalizer = self.ntokens if self.ntokens is not None else self.nseqs
        elif normalization == "none":
            normalizer = 1
        else:
            raise ValueError(f"Unknown normalization option: {normalization}")

        norm_tensor = tensor / normalizer

        if n_gpu > 1:
            norm_tensor = norm_tensor / n_gpu

        if n_accumulation > 1:
            norm_tensor = norm_tensor / n_accumulation

        return norm_tensor

    def sort_by_src_length(self) -> List[int]:
        """
        Sort by src length (descending) and return index to revert sort.
        Important for RNN-based models.

        :return: list of indices for reverting the sorting
        """
        _, perm_index = self.src_length.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.src = self.src[perm_index]
        self.src_length = self.src_length[perm_index]
        self.src_mask = self.src_mask[perm_index]
        self.indices = self.indices[perm_index]

        if self.src_prompt_mask is not None:
            self.src_prompt_mask = self.src_prompt_mask[perm_index]

        if self.has_trg:
            self.trg = self.trg[perm_index]
            self.trg_mask = self.trg_mask[perm_index]
            if self.trg_prompt_mask is not None:
                self.trg_prompt_mask = self.trg_prompt_mask[perm_index]

        return rev_index

    @staticmethod
    def score(log_probs: Tensor, trg: Tensor, pad_index: int) -> np.ndarray:
        """
        Calculate scores for each target token in the batch.

        :param log_probs: model output logits [batch_size, seq_len, vocab_size]
        :param trg: target tokens [batch_size, seq_len]
        :param pad_index: index of padding token
        :return: array of scores for each sequence
        """
        assert log_probs.size(0) == trg.size(0)
        scores = []
        for i in range(log_probs.size(0)):
            scores.append(
                np.array([
                    log_probs[i, j, ind].item() for j, ind in enumerate(trg[i])
                    if ind != pad_index
                ])
            )
        return np.array(scores, dtype=object)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nseqs={self.nseqs}, "
            f"ntokens={self.ntokens}, "
            f"has_trg={self.has_trg}, is_train={self.is_train})"
        )