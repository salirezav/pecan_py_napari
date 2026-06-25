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
    """Return an upsampled cosine-similarity heatmap (legacy / debug visualization)."""
    sweep = compute_similarity_sweep(
        model,
        frame,
        anchor_y,
        anchor_x,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        device=device,
    )
    return upsample_similarity_grid(
        sweep["sim_grid"],
        sweep["ys"],
        sweep["xs"],
        sweep["shape_hw"][0],
        sweep["shape_hw"][1],
    )


def compute_similarity_sweep(
    model: PatchEmbedder,
    frame: "np.ndarray",
    anchor_y: int,
    anchor_x: int,
    patch_size: int = 8,
    stride: int = 4,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> dict:
    """Sweep patch centers and return a sparse similarity grid vs. the anchor patch.

    Returns a dict with keys ``sim_grid`` (gh, gw), ``ys``, ``xs``, ``anchor_sim``,
    and ``shape_hw`` (H, W).
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
        anchor_sim = float((anchor_emb * anchor_emb).sum().item())

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
    return {
        "sim_grid": sim_grid,
        "ys": ys,
        "xs": xs,
        "anchor_sim": anchor_sim,
        "shape_hw": (H, W),
        "anchor_y": ay,
        "anchor_x": ax,
    }


def upsample_similarity_grid(
    sim_grid: "np.ndarray",
    ys: list,
    xs: list,
    height: int,
    width: int,
) -> "np.ndarray":
    """Bilinear upsample a patch-center similarity grid to full frame size."""
    import numpy as np
    from torch.nn.functional import interpolate as _interp

    sim_tensor = torch.from_numpy(np.asarray(sim_grid)).unsqueeze(0).unsqueeze(0)
    sim_full = _interp(
        sim_tensor,
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )
    return sim_full.squeeze().numpy()


def resolve_similarity_cutoff(sim_grid: "np.ndarray", mode: str, value: float) -> float:
    """Convert UI threshold settings to a cosine-similarity cutoff."""
    import numpy as np

    flat = np.asarray(sim_grid, dtype=np.float32).ravel()
    if flat.size == 0:
        return 1.0
    peak = float(flat.max())
    if mode == "peak_fraction":
        return peak * float(value)
    return float(value)


def build_patch_highlight_mask(
    sim_grid: "np.ndarray",
    ys: list,
    xs: list,
    patch_size: int,
    height: int,
    width: int,
    cutoff: float,
) -> "np.ndarray":
    """Highlight only stride-aligned patches whose similarity meets *cutoff*."""
    import numpy as np

    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    half = patch_size // 2
    grid = np.asarray(sim_grid)
    for gi, y in enumerate(ys):
        for gj, x in enumerate(xs):
            if grid[gi, gj] < cutoff:
                continue
            y0 = max(0, int(y) - half)
            y1 = min(int(height), int(y) - half + patch_size)
            x0 = max(0, int(x) - half)
            x1 = min(int(width), int(x) - half + patch_size)
            mask[y0:y1, x0:x1] = 1
    return mask


def similarity_mask_from_frame(
    model: PatchEmbedder,
    frame: "np.ndarray",
    anchor_y: int,
    anchor_x: int,
    *,
    patch_size: int = 8,
    stride: int = 4,
    threshold_mode: str = "peak_fraction",
    threshold_value: float = 0.92,
    device: torch.device | None = None,
) -> tuple["np.ndarray", dict]:
    """Compute a patch-highlight mask and summary stats for one frame."""
    import numpy as np

    sweep = compute_similarity_sweep(
        model,
        frame,
        anchor_y,
        anchor_x,
        patch_size=patch_size,
        stride=stride,
        device=device,
    )
    cutoff = resolve_similarity_cutoff(
        sweep["sim_grid"], threshold_mode, threshold_value
    )
    h, w = sweep["shape_hw"]
    mask = build_patch_highlight_mask(
        sweep["sim_grid"],
        sweep["ys"],
        sweep["xs"],
        patch_size,
        h,
        w,
        cutoff,
    )
    grid = np.asarray(sweep["sim_grid"])
    stats = {
        "anchor_y": int(sweep["anchor_y"]),
        "anchor_x": int(sweep["anchor_x"]),
        "anchor_sim": float(sweep["anchor_sim"]),
        "peak_sim": float(grid.max()) if grid.size else 0.0,
        "cutoff": float(cutoff),
        "patches_total": int(grid.size),
        "patches_highlighted": int(np.count_nonzero(grid >= cutoff)),
    }
    return mask, stats


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
