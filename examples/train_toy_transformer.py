from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn.functional as F

from torch_speedkit.config import SpeedConfig
from torch_speedkit.speed import apply_speedups
from torch_speedkit.trainer import Trainer
from torch_speedkit.utils import maybe_channels_last
from torch_speedkit.logging import setup_logger

log = setup_logger()


class ToyTransformer(torch.nn.Module):
    """
    Small transformer-ish block that uses PyTorch's attention ops.
    If your real model uses scaled_dot_product_attention internally,
    the SDPA backend toggles can route to Flash/mem-efficient implementations on supported GPUs.
    """
    def __init__(self, vocab: int = 32000, d_model: int = 768, n_heads: int = 12, seq: int = 256):
        super().__init__()
        self.vocab = vocab
        self.seq = seq
        self.d_model = d_model
        self.n_heads = n_heads

        self.emb = torch.nn.Embedding(vocab, d_model)
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.proj = torch.nn.Linear(d_model, d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, S]
        x = self.emb(tokens)  # [B, S, D]
        x = self.ln1(x)

        qkv = self.qkv(x)  # [B,S,3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to [B, H, S, Dh]
        b, s, d = q.shape
        h = self.n_heads
        dh = d // h

        q = q.view(b, s, h, dh).transpose(1, 2)
        k = k.view(b, s, h, dh).transpose(1, 2)
        v = v.view(b, s, h, dh).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)  # [B,H,S,Dh]
        attn = attn.transpose(1, 2).contiguous().view(b, s, d)  # [B,S,D]
        x = x + self.proj(attn)

        x = self.ln2(x)
        x = x + self.ff(x)
        logits = self.head(x)  # [B,S,V]
        return logits


def make_batch(batch_size: int, device: torch.device, seq: int, vocab: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, vocab, (batch_size, seq), device=device)
    # Next-token prediction (shifted)
    target = tokens.clone()
    return tokens, target


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # logits [B,S,V], target [B,S]
    b, s, v = logits.shape
    return F.cross_entropy(logits.view(b * s, v), target.view(b * s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = SpeedConfig.from_yaml(args.config)

    model = ToyTransformer()
    model, ctx = apply_speedups(model, cfg)

    def _make(bs: int, dev: torch.device):
        return make_batch(bs, dev, seq=model.seq, vocab=model.vocab)

    trainer = Trainer(model=model, ctx=ctx, loss_fn=loss_fn, make_batch=_make)
    log.info("Starting toy training...")
    trainer.train()


if __name__ == "__main__":
    main()
