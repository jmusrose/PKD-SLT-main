import torch
from torch import Tensor
from typing import Optional
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
        self.src_mask = (src_length.unsqueeze(1) > torch.arange(src.size(1)).to(src.device))
        self.src_prompt_mask = src_prompt_mask
        self.trg = None
        self.trg_mask = None
        self.trg_prompt_mask = None
        self.indices = indices

        self.nseqs = src.size(0)
        self.has_trg = trg is not None
        self.is_train = is_train

        if self.has_trg:
            self.trg = trg
            self.trg_mask = (self.trg != pad_index).unsqueeze(1)
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nseqs={self.nseqs}, "
            f"has_trg={self.has_trg}, is_train={self.is_train})"
        )