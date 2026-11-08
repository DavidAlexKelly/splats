"""
Per-photo appearance correction with bilateral grids (as in gsplat's examples / "Bilateral
Guided Radiance Field Processing", Wang et al. 2024).

Each training photo gets a small 3D grid (x, y, brightness) of 3x4 colour transforms. The
splat is rendered, then each pixel is corrected by the transform looked up at its position and
brightness, before comparing with the photo. The grids soak up what the scene itself shouldn't
have to explain: auto exposure, white balance drift, vignetting, local tone mapping, the warm
interior light vs the blue garden in the same frame. The splat keeps the consistent colours;
the grids are not exported, so every viewer shows the corrected, consistent scene.

A total-variation term keeps each grid smooth, so it can only make gentle, low-frequency
changes and never paint detail in, and recenter_() keeps the corrections averaging to "no
change", so the splat itself holds the true average colours.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ITU-R BT.601 luma, used as the "brightness" axis of the grid
_LUMA = (0.299, 0.587, 0.114)


class BilateralGrids(torch.nn.Module):
    def __init__(self, num_images: int, grid_xy: int = 16, grid_luma: int = 8):
        super().__init__()
        eye = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])  # identity
        grids = eye.reshape(1, 12, 1, 1, 1).repeat(num_images, 1, grid_luma, grid_xy, grid_xy)
        self.grids = torch.nn.Parameter(grids)  # [N, 12, L, Y, X]
        self.register_buffer("luma", torch.tensor(_LUMA).view(1, 1, 1, 3))

    def forward(self, rgb: torch.Tensor, idx: int) -> torch.Tensor:
        """rgb [1, H, W, 3] in 0..1 -> corrected [1, H, W, 3] using photo idx's grid."""
        _, H, W, _ = rgb.shape
        dev = rgb.device
        ys = torch.linspace(-1, 1, H, device=dev)
        xs = torch.linspace(-1, 1, W, device=dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        lum = (rgb * self.luma).sum(-1)[0].clamp(0, 1) * 2 - 1  # [H, W]
        coords = torch.stack([gx, gy, lum], -1)[None, None]  # [1, 1, H, W, 3] = (x, y, z)
        coef = F.grid_sample(self.grids[idx:idx + 1], coords, mode="bilinear",
                             padding_mode="border", align_corners=True)  # [1, 12, 1, H, W]
        A = coef[0, :, 0].permute(1, 2, 0).reshape(H, W, 3, 4)
        out = (A[..., :3] @ rgb[0, ..., None]).squeeze(-1) + A[..., 3]
        return out[None]

    @torch.no_grad()
    def recenter_(self, idx: list[int] | torch.Tensor):
        """Keep the average correction over the training photos at exactly 'no change'.

        Without this the corrections and the splat can drift together (the splat turns purple
        and every photo's grid turns it back), which looks fine in training but wrong in any
        viewer and in Fixer, which see the splat without corrections."""
        g = self.grids[idx]
        eye = torch.tensor([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0], device=g.device)
        drift = g.mean(0, keepdim=True) - eye.view(1, 12, 1, 1, 1)
        self.grids[idx] = g - drift

    def tv_loss(self, idx: int | None = None) -> torch.Tensor:
        """Smoothness of the grid(s) along x, y and brightness."""
        g = self.grids if idx is None else self.grids[idx:idx + 1]
        return ((g[:, :, 1:] - g[:, :, :-1]).pow(2).mean()
                + (g[:, :, :, 1:] - g[:, :, :, :-1]).pow(2).mean()
                + (g[:, :, :, :, 1:] - g[:, :, :, :, :-1]).pow(2).mean())


def make(num_images: int, s: dict, device):
    """(grids, optimizer) from settings, or (None, None) when bilagrid is off."""
    if not s.get("bilagrid"):
        return None, None
    grids = BilateralGrids(num_images, int(s["bilagrid_xy"]), int(s["bilagrid_luma"])).to(device)
    opt = torch.optim.Adam(grids.parameters(), lr=float(s["bilagrid_lr"]), eps=1e-15)
    return grids, opt
