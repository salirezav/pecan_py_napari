"""Contrastive embedding model – architecture from Pecan Contrastive Coding.

Lightweight MobileNet-style encoder (MBConv + Squeeze-Excite) that maps
small image patches to L2-normalised 64-D embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthwiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, stride=1, bias=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, stride=1, bias=bias, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


class _SqueezeExcite(nn.Module):
    def __init__(self, ch, reduction=2):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // reduction)
        self.fc2 = nn.Linear(ch // reduction, ch)

    def forward(self, x):
        b, c, h, w = x.shape
        s = F.avg_pool2d(x, (h, w)).reshape(b, -1)
        s = F.leaky_relu(self.fc1(s))
        s = F.leaky_relu(self.fc2(s)).reshape(b, c, 1, 1)
        return x * s


class _MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand=2, ks=5, pad=2, dropout=True, bias=False):
        super().__init__()
        mid = in_ch * expand
        self.expand = nn.Conv2d(in_ch, mid, 1, bias=bias)
        self.dw = _DepthwiseConv(mid, mid, ks, padding=pad, bias=bias)
        self.gn = nn.GroupNorm(expand, mid)
        self.se = _SqueezeExcite(mid)
        self.project = nn.Conv2d(mid, out_ch, 1, bias=bias)
        self.dropout = dropout

    def forward(self, x):
        h = self.expand(x)
        h = self.gn(self.dw(h))
        h = F.silu(self.se(h + self.expand(x)))
        if self.dropout and self.training:
            h = F.dropout(h, p=0.3)
        return self.project(h)


class PatchEmbedder(nn.Module):
    """Maps (B, C, ps, ps) patches to L2-normalised (B, 64) embeddings."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.stage1 = nn.Sequential(_MBConv(in_channels, 32), _MBConv(32, 32), _MBConv(32, 32))
        self.stage2 = nn.Sequential(_MBConv(32, 64), _MBConv(64, 64), _MBConv(64, 64))
        self.head = nn.Sequential(nn.Linear(64, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = F.avg_pool2d(x, 2, 2)
        x = self.stage2(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        h = self.head[1](self.head[0](x))
        out = self.head[3](self.head[2](h)) + h
        return F.normalize(out, p=2, dim=1)


def compute_similarity_map(
    model: PatchEmbedder,
    frame: "np.ndarray",
    anchor_y: int,
    anchor_x: int,
    patch_size: int = 8,
    stride: int = 4,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> "np.ndarray":
    """Sweep the frame and return a cosine-similarity heatmap vs. the anchor.

    Parameters
    ----------
    model : trained PatchEmbedder (eval mode, on *device*)
    frame : (H, W, C) uint8 image
    anchor_y, anchor_x : centre of the anchor patch (pixel coords)
    patch_size, stride : sweep parameters
    batch_size : patches per forward pass
    device : torch device (defaults to model's device)

    Returns
    -------
    sim_map : (H, W) float32 array with values in [-1, 1]
    """
    import numpy as np

    if device is None:
        device = next(model.parameters()).device

    H, W = frame.shape[:2]
    C = frame.shape[2] if frame.ndim == 3 else 1
    half = patch_size // 2

    ay = max(half, min(anchor_y, H - half - 1))
    ax = max(half, min(anchor_x, W - half - 1))
    anchor_patch = frame[ay - half : ay - half + patch_size,
                         ax - half : ax - half + patch_size]
    if anchor_patch.ndim == 2:
        anchor_patch = anchor_patch[..., np.newaxis]
    anchor_t = (
        torch.from_numpy(anchor_patch.transpose(2, 0, 1).astype(np.float32) / 255.0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        anchor_emb = model(anchor_t)

    ys = list(range(half, H - half, stride))
    xs = list(range(half, W - half, stride))
    grid_h, grid_w = len(ys), len(xs)

    coords = [(y, x) for y in ys for x in xs]
    all_sims = np.empty(len(coords), dtype=np.float32)

    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start : start + batch_size]
        patches = np.empty((len(batch_coords), C, patch_size, patch_size), dtype=np.float32)
        for i, (cy, cx) in enumerate(batch_coords):
            p = frame[cy - half : cy - half + patch_size,
                      cx - half : cx - half + patch_size]
            if p.ndim == 2:
                p = p[..., np.newaxis]
            patches[i] = p.transpose(2, 0, 1).astype(np.float32) / 255.0

        with torch.no_grad():
            embs = model(torch.from_numpy(patches).to(device))
            sims = (embs * anchor_emb).sum(dim=1).cpu().numpy()
        all_sims[start : start + len(batch_coords)] = sims

    sim_grid = all_sims.reshape(grid_h, grid_w)

    from torch.nn.functional import interpolate as _interp
    sim_tensor = torch.from_numpy(sim_grid).unsqueeze(0).unsqueeze(0)
    sim_full = _interp(sim_tensor, size=(H, W), mode="bilinear", align_corners=False)
    return sim_full.squeeze().numpy()


def contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """NT-Xent–style loss.

    anchor, positive : (N, D)
    negatives        : (N, K, D)  – K negatives per anchor
    """
    pos_sim = (anchor * positive).sum(dim=1, keepdim=True) / temperature
    neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)
