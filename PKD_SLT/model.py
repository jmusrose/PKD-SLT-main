

from typing import Dict
from PKD_SLT.vocabulary import Vocabulary
from PKD_SLT.helpers_for_ddp import get_logger
from PKD_SLT.embeddings import Sign_Embedding, Embeddings

logging = get_logger(__name__)

def build_model(
    cfg: Dict = None,
    sign_dim: int = 783,
    trg_vocab: Vocabulary = None
):
    logging.info("正在构建编码器解码器")
    enc_cfg = cfg["encoder"]
    dec_cfg = cfg["decoder"]
    trg_pad_index = trg_vocab.pad_index
    # 手语特征嵌入
    sgn_embed = Sign_Embedding(
        **enc_cfg["embedding"],
        num_heads=enc_cfg["num_heads"],
        input_size= sign_dim
    )
    # 解码器嵌入
    trg_embed = Embeddings(
        **dec_cfg["embeddings"],
        vocab_size=len(trg_vocab),
        num_heads=dec_cfg["num_heads"],
        padding_idx=trg_pad_index,
    )

    enc_dropout = enc_cfg.get("dropout", 0.0)
    enc_emb_dropout = enc_cfg["embeddings"].get("dropout", enc_dropout)
    if enc_cfg["type"] == "transformer":
        assert enc_cfg["embeddings"]["embedding_dim"] == enc_cfg["hidden_size"], (
            "transformer输入维度和嵌入维度必须相同"
        )
        emb_size = sgn_embed.embedding_dim



