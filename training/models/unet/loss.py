"""
Loss Functions for Gridded PM2.5 U-Net (v3)
============================================

StationListLoss = bilinear-sampled station MAE + lambda_tv * TV smoothness.

Key change from v2: uses F.grid_sample for differentiable bilinear interpolation
at exact station coordinates instead of integer pixel indexing. Eliminates
discretization error and provides smoother gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StationListLoss(nn.Module):
    """
    Combined loss for sparse-supervised PM2.5 grid prediction.

    Components:
        1. Bilinear-sampled MAE: use grid_sample to interpolate predicted map
           at exact station locations, compute MAE against observed PM2.5.
        2. Total Variation (TV): spatial smoothness inside Pakistan border.

    Parameters
    ----------
    lambda_tv : float
        Weight for the TV smoothness term (default 0.01).
    """

    def __init__(self, lambda_tv: float = 0.01):
        super().__init__()
        self.lambda_tv = lambda_tv

    def forward(
        self,
        pred: torch.Tensor,
        station_grids: list,
        station_pm25: list,
        border: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:          [B, 1, H_out, W_out] predicted PM2.5 at 0.05°
            station_grids: list of B tensors, each [N_i, 2] (x, y) in [-1,1]
            station_pm25:  list of B tensors, each [N_i] float32 observed PM2.5
            border:        [B, 1, H_in, W_in] Pakistan boundary at 0.1°

        Returns:
            Scalar loss.
        """
        # Bilinear-sampled station MAE
        all_errors = []
        for b in range(pred.shape[0]):
            coords = station_grids[b]  # [N, 2]
            obs = station_pm25[b]      # [N]
            if len(coords) == 0:
                continue
            # grid_sample expects [B, 1, N, 2] grid → [B, 1, N, 1] output
            grid = coords.to(pred.device).unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
            sampled = F.grid_sample(
                pred[b : b + 1], grid,
                mode="bilinear", align_corners=True, padding_mode="zeros",
            )  # [1, 1, N, 1]
            pred_at_stations = sampled[0, 0, :, 0]  # [N]
            all_errors.append(torch.abs(pred_at_stations - obs.to(pred.device)))

        if all_errors:
            mae = torch.cat(all_errors).mean()
        else:
            mae = torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Total Variation smoothness inside border
        if self.lambda_tv > 0:
            border_hi = F.interpolate(
                border, size=pred.shape[2:], mode="nearest"
            )
            tv = self._tv_loss(pred, border_hi)
            return mae + self.lambda_tv * tv
        return mae

    @staticmethod
    def _tv_loss(pred: torch.Tensor, border: torch.Tensor) -> torch.Tensor:
        """Anisotropic total variation inside the border mask."""
        diff_h = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        border_h = border[:, :, :, 1:] * border[:, :, :, :-1]

        diff_v = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        border_v = border[:, :, 1:, :] * border[:, :, :-1, :]

        n_h = border_h.sum().clamp(min=1.0)
        n_v = border_v.sum().clamp(min=1.0)

        tv = (diff_h * border_h).sum() / n_h + (diff_v * border_v).sum() / n_v
        return tv
