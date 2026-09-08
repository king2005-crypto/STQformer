"""Losses and RGB [0,1] full-reference metrics."""
import torch
from torch import nn
from torch.nn import functional as F


def charbonnier(x, y, epsilon=1e-3):
    return ((x - y).square() + epsilon ** 2).sqrt().mean()


def ssim_per_image(x, y):
    """11x11 Gaussian, sigma=1.5, VALID window, RGB channel mean.

    No large denominator epsilon: it biases SSIM for low-variance images.
    """
    if x.shape != y.shape or min(x.shape[-2:]) < 11:
        raise ValueError("SSIM needs equal BCHW tensors with H,W >= 11")
    x, y = x.float(), y.float()
    coords = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    g = torch.exp(-coords.square() / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(x.shape[1], 1, 11, 11).contiguous()
    def filt(z):
        return F.conv2d(z, window, groups=x.shape[1])
    mx, my = filt(x), filt(y)
    vx, vy = (filt(x * x) - mx * mx).clamp_min(0), (filt(y * y) - my * my).clamp_min(0)
    cov = filt(x * y) - mx * my
    score = ((2 * mx * my + .01 ** 2) * (2 * cov + .03 ** 2)) / (
        (mx.square() + my.square() + .01 ** 2) * (vx + vy + .03 ** 2))
    return score.mean(dim=(1, 2, 3))


def temporal_consistency(pred, prev_pred, center, prev_center):
    """Residual-difference temporal consistency; both outputs get gradients.

    This computes
    mean |(O_t-O_{t-1})-(I_t-I_{t-1})| on true adjacent sliding windows.
    """
    return ((pred - prev_pred) - (center - prev_center)).abs().mean()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        self.features.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x, y = (x - self.mean) / self.std, (y - self.mean) / self.std
        loss = x.new_zeros(())
        for index, layer in enumerate(self.features):
            x, y = layer(x), layer(y)
            if index in (3, 8, 15):  # relu1_2, relu2_2, relu3_3
                loss = loss + F.l1_loss(x, y)
        return loss / 3


class SpatialLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.perceptual = PerceptualLoss() if config["w_perc"] > 0 else None

    def forward(self, pred, target):
        loss = self.config["w_pix"] * charbonnier(pred, target)
        if self.config["w_ssim"]:
            loss = loss + self.config["w_ssim"] * (1 - ssim_per_image(pred, target).mean())
        if self.perceptual is not None:
            loss = loss + self.config["w_perc"] * self.perceptual(pred, target)
        return loss
