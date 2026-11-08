import argparse
import glob
import json
import os
import time
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Model Architecture: B-IFVNet -----------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv(x)
        w = self.se(res)
        return self.relu(x + res * w)

class HesitationCalibratedGate(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch + 1, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid()
        )

    def forward(self, skip_feat, hesitation_map):
        hes_down = F.interpolate(hesitation_map, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
        weight = self.gate_conv(torch.cat([skip_feat, hes_down], dim=1))
        calibrated = skip_feat * (1.0 - 0.5 * hes_down) * weight
        return calibrated

class BiDirectionalIFVNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, c[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True)
        )
        self.enc1 = ResidualBlock(c[0])
        self.down1 = nn.Conv2d(c[0], c[1], 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(c[1])
        self.down2 = nn.Conv2d(c[1], c[2], 3, stride=2, padding=1)
        self.enc3 = ResidualBlock(c[2])
        self.down3 = nn.Conv2d(c[2], c[3], 3, stride=2, padding=1)
        self.bottleneck = ResidualBlock(c[3])

        self.gate3 = HesitationCalibratedGate(c[2])
        self.gate2 = HesitationCalibratedGate(c[1])
        self.gate1 = HesitationCalibratedGate(c[0])

        self.aux_head = nn.Conv2d(c[3], 1, 1)

        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c[2] * 2, c[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            ResidualBlock(c[2])
        )

        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c[1] * 2, c[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            ResidualBlock(c[1])
        )

        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c[0] * 2, c[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            ResidualBlock(c[0])
        )
        self.out_seg = nn.Conv2d(c[0], 1, 1)

    def forward(self, x, acoustic_grad=None):
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        aux_logit = self.aux_head(b)
        aux_prob = torch.sigmoid(F.interpolate(aux_logit, size=x.shape[-2:], mode='bilinear', align_corners=False))
        p_c = aux_prob.clamp(1e-6, 1.0 - 1e-6)
        entropy = - (p_c * torch.log2(p_c) + (1.0 - p_c) * torch.log2(1.0 - p_c))
        hesitation = torch.tanh(1.5 * entropy + 0.5 * acoustic_grad) if acoustic_grad is not None else torch.tanh(1.5 * entropy)

        e3_cal = self.gate3(e3, hesitation)
        d3 = self.dec3(torch.cat([self.up3(b), e3_cal], dim=1))

        e2_cal = self.gate2(e2, hesitation)
        d2 = self.dec2(torch.cat([self.up2(d3), e2_cal], dim=1))

        e1_cal = self.gate1(e1, hesitation)
        d1 = self.dec1(torch.cat([self.up1(d2), e1_cal], dim=1))

        final_logit = self.out_seg(d1)
        return final_logit, aux_logit, hesitation

# ----------------- Bi-directional IFV Loss -----------------
class BiDirectionalIFVLoss(nn.Module):
    def __init__(self, lambda_hes=0.5, eps=1e-6):
        super().__init__()
        self.lambda_hes = lambda_hes
        self.eps = eps

    def forward(self, final_logit, aux_logit, targets, hesitation):
        prob = torch.sigmoid(final_logit)
        certainty = (1.0 - hesitation).clamp(min=0.0, max=1.0)

        bce = F.binary_cross_entropy_with_logits(final_logit, targets, reduction='none')
        l_bce = (certainty * bce).sum() / (certainty.sum() + self.eps)

        inter = (prob * targets * certainty).sum(dim=(-2, -1))
        union = (prob * certainty).sum(dim=(-2, -1)) + (targets * certainty).sum(dim=(-2, -1))
        l_dice = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()

        kernel = torch.ones((1, 1, 3, 3), device=final_logit.device) / 9.0
        smooth_prob = F.conv2d(prob.detach(), kernel, padding=1)
        l_hes = (hesitation * (prob - smooth_prob) ** 2).sum() / (hesitation.sum() + self.eps)

        aux_up = F.interpolate(aux_logit, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        l_aux = F.binary_cross_entropy_with_logits(aux_up, targets)
        return l_bce + l_dice + self.lambda_hes * l_hes + 0.3 * l_aux

def compute_acoustic_gradient(img):
    gx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    gy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    local_var = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return local_var / (local_var.amax(dim=(-2, -1), keepdim=True) + 1e-6)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="B-IFVNet")
    p.add_argument("--data", default="/root/autodl-tmp/ssl_fuzzy_seg/data/busi_raw")
    p.add_argument("--out", default="/root/autodl-tmp/ifv_seg/results_v2")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.5)
    p.add_argument("--lambda_hes", type=float, default=0.5)
    args = p.parse_args()
