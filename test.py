import math
import torch
import torch.nn as nn
from PKD_SLT.embeddings import Embeddings     # ← 保证已 import 你的实现

# ----------------- 超参数 -----------------
vocab_size   = 1000         # 词表大小
embedding_dim = 64          # d_model
batch_size   = 2
seq_len      = 10
num_heads    = 8

# ----------------- 构造模型 -----------------
emb = Embeddings(
    embedding_dim   = embedding_dim,
    num_heads       = num_heads,
    scale           = True,
    norm_type       = "batch",      # "batch" | "group" | "layer" | None
    activation_type = "relu",       # "relu" | "gelu"  | None
    vocab_size      = vocab_size,
    padding_idx     = 0,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
emb.to(device)

print(emb)                            # __repr__ 检查

# ----------------- 造假数据 -----------------
torch.manual_seed(0)
ids  = torch.randint(1,   (batch_size, seq_len), device=device)

# 假设第 2 个样本只有前 6 个 token 有效，后 4 个是 <pad>(id=0)
ids[1, 6:] = 0

mask = (ids != 0)                     # (B, L)  bool

# ----------------- 前向 -----------------
emb.train()                           # 训练模式：用 MaskedNorm
out = emb(ids, mask)                  # (B, L, embedding_dim)

print("输出形状:", out.shape)          # torch.Size([2, 10, 64])

# --------- 检查归一化 (仅有效 token) ---------
valid = out[mask]                     # 取出 mask==True 的位置
print("有效位置均值≈0 :", valid.mean(0)[:4])     # 取前4维示例
print("有效位置方差≈1 :", valid.std(0,unbiased=False)[:4])

# --------- 反向传播 ----------
loss = out.mean()
loss.backward()
print("grad 是否存在:", emb.lut.weight.grad is not None)

# --------- 推理模式 ----------
emb.eval()
with torch.no_grad():
    out_eval = emb(ids, mask)
print("eval 形状:", out_eval.shape)
