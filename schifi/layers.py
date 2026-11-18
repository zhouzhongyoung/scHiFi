from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZINBLoss(nn.Module):
    """Negative log-likelihood of a zero-inflated negative-binomial model."""

    def forward(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        disp: torch.Tensor,
        pi: torch.Tensor,
        scale_factor: torch.Tensor,
        ridge_lambda: float = 0.0,
    ) -> torch.Tensor:
        eps = 1e-10
        scale_factor = scale_factor.reshape(-1, 1)
        mean = mean * scale_factor

        t1 = (
            torch.lgamma(disp + eps)
            + torch.lgamma(x + 1.0)
            - torch.lgamma(x + disp + eps)
        )
        t2 = (disp + x) * torch.log1p(mean / (disp + eps)) + x * (
            torch.log(disp + eps) - torch.log(mean + eps)
        )
        nb_case = t1 + t2 - torch.log(1.0 - pi + eps)

        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
        result = torch.where(x <= 1e-8, zero_case, nb_case)

        if ridge_lambda > 0:
            result = result + ridge_lambda * pi.square()
        return result.mean()


class MeanAct(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
