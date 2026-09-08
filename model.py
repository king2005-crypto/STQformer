"""STQFormer example: frame-wise encoding, STQT blocks, and center-frame decoding."""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class QuaternionInspiredSpatial(nn.Module):
    """Eq. (2): pointwise mixing -> depthwise 3x3 -> pointwise projection.

    Channels are contiguous groups of four. Dense pointwise layers mix both
    components and groups. This is NOT a Hamilton-product convolution: the
    spatial block uses the pointwise-depthwise-pointwise form of Eq. (2).
    """
    def __init__(self, channels):
        super().__init__()
        if channels % 4:
            raise ValueError("Quaternion feature channels must be divisible by 4")
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels, 1)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x + self.pw2(self.dw(self.pw1(self.norm(x))))


class STQTBlock(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.spatial = QuaternionInspiredSpatial(channels)
        self.temporal_norm = nn.LayerNorm(channels)
        self.temporal = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x, use_temporal=True):
        b, t, c, h, w = x.shape
        x = self.spatial(x.reshape(b * t, c, h, w)).reshape(b, t, c, h, w)
        if use_temporal:
            tokens = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
            z = self.temporal_norm(tokens)
            tokens = tokens + self.temporal(z, z, z, need_weights=False)[0]
            x = tokens.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2)
        return x


class RestormerBlock(nn.Module):
    """Compact independently implemented MDTA + gated depthwise FFN.

    Channel attention (C/head x C/head), not quadratic spatial attention.
    See the Restormer paper and the attribution in THIRD_PARTY.md.
    """
    def __init__(self, channels, heads=4, expansion=2.66):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = LayerNorm2d(channels), LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.qkv_dw = nn.Conv2d(3 * channels, 3 * channels, 3, padding=1,
                                groups=3 * channels, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.out = nn.Conv2d(channels, channels, 1, bias=False)
        hidden = int(channels * expansion)
        self.ff_in = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.ff_dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                              groups=hidden * 2, bias=False)
        self.ff_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(self.norm1(x))).chunk(3, dim=1)
        q, k, v = [a.reshape(b, self.heads, c // self.heads, h * w) for a in (q, k, v)]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attention = ((q @ k.transpose(-1, -2)) * self.temperature).softmax(dim=-1)
        x = x + self.out((attention @ v).reshape(b, c, h, w))
        a, gate = self.ff_dw(self.ff_in(self.norm2(x))).chunk(2, dim=1)
        return x + self.ff_out(F.gelu(a) * gate)


class CenterDecoder(nn.Module):
    """Restormer-style feature decoder; not the complete Restormer U-Net."""
    def __init__(self, channels, width, blocks, heads):
        super().__init__()
        self.proj = nn.Conv2d(channels, width, 1)
        self.blocks = nn.Sequential(*[RestormerBlock(width, heads) for _ in range(blocks)])
        self.rgb = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, feature, size):
        x = self.rgb(self.blocks(self.proj(feature)))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class SwinEncoder(nn.Module):
    def __init__(self, config, initialize_pretrained):
        super().__init__()
        from transformers import SwinConfig, SwinModel
        if config["encoder"] == "swin_toy":
            # Actual Swin implementation, small RANDOM weights for offline tests.
            swin_config = SwinConfig(image_size=config["image_size"], patch_size=4,
                                     embed_dim=16, depths=[1, 1], num_heads=[2, 4],
                                     window_size=2, drop_path_rate=0.0)
            self.backbone = SwinModel(swin_config, add_pooling_layer=False)
        elif config["encoder"] == "swin":
            if "swin_config" in config:
                self.backbone = SwinModel(SwinConfig.from_dict(config["swin_config"]),
                                          add_pooling_layer=False)
            elif initialize_pretrained:
                self.backbone = SwinModel.from_pretrained(
                    config["encoder_name"], add_pooling_layer=False)
            else:
                raise ValueError("Checkpoint must contain swin_config for offline restoration")
        else:
            raise ValueError("encoder must be swin or swin_toy")
        config["swin_config"] = self.backbone.config.to_dict()
        self.proj = nn.Conv2d(self.backbone.config.hidden_size, config["embed_dim"], 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, frames):
        result = self.backbone(pixel_values=(frames - self.mean) / self.std,
                               output_hidden_states=True)
        # HF exposes the actual H,W; do not guess a square from token count.
        return self.proj(result.reshaped_hidden_states[-1])


class STQFormer(nn.Module):
    def __init__(self, config, initialize_pretrained=True):
        super().__init__()
        self.config = dict(config)
        c, heads = config["embed_dim"], config["heads"]
        if c % 4 or c % heads or config["decoder_dim"] % heads:
            raise ValueError("embed_dim must divide by 4 and heads; decoder_dim by heads")
        self.encoder = SwinEncoder(self.config, initialize_pretrained)
        self.blocks = nn.ModuleList([STQTBlock(c, heads) for _ in range(config["num_blocks"])])
        self.decoder = CenterDecoder(c, config["decoder_dim"], config["decoder_blocks"], heads)
        self.stage = "full"

    def set_stage(self, stage):
        if stage not in ("spatial", "full"):
            raise ValueError(stage)
        self.stage = stage
        # Freeze the pretrained backbone AND projection in the spatial stage.
        for p in self.encoder.parameters():
            p.requires_grad_(stage == "full")
        for block in self.blocks:
            for module in (block.temporal, block.temporal_norm):
                for p in module.parameters():
                    p.requires_grad_(stage == "full")
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.stage == "spatial":
            self.encoder.eval()  # Also disable pretrained stochastic depth/dropout.
        return self

    def forward(self, clip):
        if clip.ndim != 5 or clip.shape[2] != 3 or clip.shape[1] % 2 != 1:
            raise ValueError("Expected [B, odd T, 3, H, W], RGB in [0,1]")
        b, t, _, h, w = clip.shape
        f = self.encoder(clip.reshape(b * t, 3, h, w))
        f = f.reshape(b, t, *f.shape[1:])
        for block in self.blocks:
            f = block(f, use_temporal=self.stage == "full")
        return self.decoder(f[:, t // 2], (h, w))
