import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)
    )

# 1. Standard U-Net
class StandardUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.e1, self.e2, self.e3, self.e4, self.e5 = conv_block(1, c[0]), conv_block(c[0], c[1]), conv_block(c[1], c[2]), conv_block(c[2], c[3]), conv_block(c[3], c[4])
        self.up4, self.d4 = nn.ConvTranspose2d(c[4], c[3], 2, 2), conv_block(c[3] * 2, c[3])
        self.up3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), conv_block(c[2] * 2, c[2])
        self.up2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), conv_block(c[1] * 2, c[1])
        self.up1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), conv_block(c[0] * 2, c[0])
        self.seg = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        e5 = self.e5(F.max_pool2d(e4, 2))
        d4 = self.d4(torch.cat([self.up4(e5), e4], 1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.seg(d1)

# 2. Attention U-Net
class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)

class AttentionUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base*2, base*4, base*8, base*16]
        self.e1, self.e2, self.e3, self.e4, self.e5 = conv_block(1, c[0]), conv_block(c[0], c[1]), conv_block(c[1], c[2]), conv_block(c[2], c[3]), conv_block(c[3], c[4])
        self.ag4 = AttentionGate(c[3], c[3], c[2])
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, 2)
        self.d4 = conv_block(c[3]*2, c[3])
        self.ag3 = AttentionGate(c[2], c[2], c[1])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.d3 = conv_block(c[2]*2, c[2])
        self.ag2 = AttentionGate(c[1], c[1], c[0])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.d2 = conv_block(c[1]*2, c[1])
        self.ag1 = AttentionGate(c[0], c[0], max(c[0]//2, 8))
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.d1 = conv_block(c[0]*2, c[0])
        self.seg = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        e5 = self.e5(F.max_pool2d(e4, 2))
        d4 = self.up4(e5)
        d4 = self.d4(torch.cat([d4, self.ag4(g=d4, x=e4)], 1))
        d3 = self.up3(d4)
        d3 = self.d3(torch.cat([d3, self.ag3(g=d3, x=e3)], 1))
        d2 = self.up2(d3)
        d2 = self.d2(torch.cat([d2, self.ag2(g=d2, x=e2)], 1))
        d1 = self.up1(d2)
        d1 = self.d1(torch.cat([d1, self.ag1(g=d1, x=e1)], 1))
        return self.seg(d1)

# 3. U-Net++
class UNetPlusPlus(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base*2, base*4, base*8]
        self.x0_0 = conv_block(1, c[0])
        self.x1_0 = conv_block(c[0], c[1])
        self.x2_0 = conv_block(c[1], c[2])
        self.x3_0 = conv_block(c[2], c[3])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.x0_1 = conv_block(c[0] + c[1], c[0])
        self.x1_1 = conv_block(c[1] + c[2], c[1])
        self.x2_1 = conv_block(c[2] + c[3], c[2])
        self.x0_2 = conv_block(c[0]*2 + c[1], c[0])
        self.x1_2 = conv_block(c[1]*2 + c[2], c[1])
        self.x0_3 = conv_block(c[0]*3 + c[1], c[0])
        self.seg = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(F.max_pool2d(x0_0, 2))
        x0_1 = self.x0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.x2_0(F.max_pool2d(x1_0, 2))
        x1_1 = self.x1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x3_0 = self.x3_0(F.max_pool2d(x2_0, 2))
        x2_1 = self.x2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        return self.seg(x0_3)

# 4. DeepLabV3+
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.out = nn.Sequential(nn.Conv2d(out_ch*4, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, x):
        return self.out(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], 1))

class DeepLabV3Plus(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = conv_block(1, base*2)
        self.enc2 = conv_block(base*2, base*4)
        self.enc3 = conv_block(base*4, base*8)
        self.aspp = ASPP(base*8, base*4)
        self.low_proj = nn.Sequential(nn.Conv2d(base*2, base*2, 1, bias=False), nn.BatchNorm2d(base*2), nn.ReLU(True))
        self.dec = nn.Sequential(conv_block(base*6, base*4), nn.Conv2d(base*4, 1, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        a = self.aspp(e3)
        a_up = F.interpolate(a, scale_factor=4, mode='bilinear', align_corners=True)
        return self.dec(torch.cat([a_up, self.low_proj(e1)], 1))

# 5. MC-Dropout
class MCDropoutUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base*2, base*4, base*8]
        self.e1 = conv_block(1, c[0])
        self.e2 = conv_block(c[0], c[1])
        self.e3 = conv_block(c[1], c[2])
        self.drop = nn.Dropout2d(p=0.2)
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.d2 = conv_block(c[1]*2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.d1 = conv_block(c[0]*2, c[0])
        self.seg = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.drop(self.e3(F.max_pool2d(e2, 2)))
        d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.seg(d1)

# 6. Mamba-UNet (2025 SOTA)
class MambaSS2DBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_in = nn.Conv2d(d_model, d_model * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(d_model * 2, d_model * 2, 3, padding=1, groups=d_model * 2, bias=False)
        self.act = nn.SiLU()
        self.proj_out = nn.Conv2d(d_model * 2, d_model, 1, bias=False)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x):
        res = x
        x1 = self.dwconv(self.proj_in(x))
        x1 = self.act(x1)
        x_out = self.norm(self.proj_out(x1))
        return res + x_out

class MambaUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base*2, base*4, base*8]
        self.e1 = conv_block(1, c[0])
        self.m1 = MambaSS2DBlock(c[0])
        self.down1 = nn.Conv2d(c[0], c[1], 3, stride=2, padding=1)
        self.e2 = conv_block(c[1], c[1])
        self.m2 = MambaSS2DBlock(c[1])
        self.down2 = nn.Conv2d(c[1], c[2], 3, stride=2, padding=1)
        self.e3 = conv_block(c[2], c[2])
        self.m3 = MambaSS2DBlock(c[2])
        self.down3 = nn.Conv2d(c[2], c[3], 3, stride=2, padding=1)
        self.b = nn.Sequential(conv_block(c[3], c[3]), MambaSS2DBlock(c[3]))
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.d3 = conv_block(c[2]*2, c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.d2 = conv_block(c[1]*2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.d1 = conv_block(c[0]*2, c[0])
        self.seg = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.m1(self.e1(x))
        e2 = self.m2(self.e2(self.down1(e1)))
        e3 = self.m3(self.e3(self.down2(e2)))
        b = self.b(self.down3(e3))
        d3 = self.d3(torch.cat([self.up3(b), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.seg(d1)

# 7. BoundaryFormer (2025 SOTA)
class BoundaryFormer(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base*2, base*4, base*8]
        self.enc1 = conv_block(1, c[0])
        self.enc2 = conv_block(c[0], c[1])
        self.enc3 = conv_block(c[1], c[2])
        self.enc4 = conv_block(c[2], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.dec3 = conv_block(c[2]*2, c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.dec2 = conv_block(c[1]*2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.dec1 = conv_block(c[0]*2, c[0])
        self.seg_head = nn.Conv2d(c[0], 1, 1)
        self.bnd_head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.seg_head(d1), self.bnd_head(d1)

# 8. UltraLight VM-UNet (2025/2026 SOTA)
class UltraLightVMUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base*2, base*4, base*8]
        self.in_conv = nn.Conv2d(1, c[0], 3, padding=1)
        self.vm1 = nn.Sequential(nn.Conv2d(c[0], c[0], 3, padding=1, groups=c[0]), nn.GELU(), nn.Conv2d(c[0], c[0], 1))
        self.down1 = nn.Conv2d(c[0], c[1], 3, stride=2, padding=1)
        self.vm2 = nn.Sequential(nn.Conv2d(c[1], c[1], 3, padding=1, groups=c[1]), nn.GELU(), nn.Conv2d(c[1], c[1], 1))
        self.down2 = nn.Conv2d(c[1], c[2], 3, stride=2, padding=1)
        self.vm3 = nn.Sequential(nn.Conv2d(c[2], c[2], 3, padding=1, groups=c[2]), nn.GELU(), nn.Conv2d(c[2], c[2], 1))
        self.down3 = nn.Conv2d(c[2], c[3], 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(nn.Conv2d(c[3], c[3], 3, padding=1, groups=c[3]), nn.GELU(), nn.Conv2d(c[3], c[3], 1))
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.dec3 = nn.Conv2d(c[2]*2, c[2], 3, padding=1)
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.dec2 = nn.Conv2d(c[1]*2, c[1], 3, padding=1)
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.dec1 = nn.Conv2d(c[0]*2, c[0], 3, padding=1)
        self.seg = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.vm1(self.in_conv(x))
        e2 = self.vm2(self.down1(e1))
        e3 = self.vm3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = F.gelu(self.dec3(torch.cat([self.up3(b), e3], 1)))
        d2 = F.gelu(self.dec2(torch.cat([self.up2(d3), e2], 1)))
        d1 = F.gelu(self.dec1(torch.cat([self.up1(d2), e1], 1)))
        return self.seg(d1)

# 9. ProtoCAM (2026 SOTA)
class ProtoCAM(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base*2, base*4, base*8]
        self.enc1 = conv_block(1, c[0])
        self.enc2 = conv_block(c[0], c[1])
        self.enc3 = conv_block(c[1], c[2])
        self.enc4 = conv_block(c[2], c[3])
        self.prototype = nn.Parameter(torch.randn(1, c[3], 1, 1))
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.dec3 = conv_block(c[2]*2, c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.dec2 = conv_block(c[1]*2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.dec1 = conv_block(c[0]*2, c[0])
        self.seg = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        proto_norm = F.normalize(self.prototype, dim=1)
        e4_norm = F.normalize(e4, dim=1)
        cam_weight = torch.sigmoid((e4_norm * proto_norm).sum(1, keepdim=True))
        e4 = e4 * (1.0 + cam_weight)
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.seg(d1)

# 10. B-IFVNet (Proposed)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(True), nn.Conv2d(channels // 4, channels, 1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        res = self.conv(x)
        return self.relu(x + res * self.se(res))

class HesitationCalibratedGate(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch + 1, in_ch, 1, bias=False), nn.BatchNorm2d(in_ch), nn.Sigmoid()
        )

    def forward(self, skip_feat, hesitation_map):
        hes_down = F.interpolate(hesitation_map, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
        weight = self.gate_conv(torch.cat([skip_feat, hes_down], dim=1))
        return skip_feat * (1.0 - 0.5 * hes_down) * weight

class BiDirectionalIFVNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.in_conv = nn.Sequential(nn.Conv2d(in_ch, c[0], 3, padding=1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(True))
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
        self.dec3 = nn.Sequential(nn.Conv2d(c[2] * 2, c[2], 3, padding=1, bias=False), nn.BatchNorm2d(c[2]), nn.ReLU(True), ResidualBlock(c[2]))
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(c[1] * 2, c[1], 3, padding=1, bias=False), nn.BatchNorm2d(c[1]), nn.ReLU(True), ResidualBlock(c[1]))
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(c[0] * 2, c[0], 3, padding=1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(True), ResidualBlock(c[0]))
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
        return self.out_seg(d1), aux_logit, hesitation

def build_model(name):
    if name == "b_ifvnet": return BiDirectionalIFVNet(1, 32)
    elif name == "standard_unet": return StandardUNet(16)
    elif name == "att_unet": return AttentionUNet(16)
    elif name == "unet_plus": return UNetPlusPlus(16)
    elif name == "deeplabv3": return DeepLabV3Plus(16)
    elif name == "fcm_seg": return StandardUNet(16)
    elif name == "frs_seg": return StandardUNet(16)
    elif name == "mc_dropout": return MCDropoutUNet(16)
    elif name == "mamba_unet": return MambaUNet(16)
    elif name == "boundary_former": return BoundaryFormer(16)
    elif name == "vm_unet": return UltraLightVMUNet(16)
    elif name == "proto_cam": return ProtoCAM(16)
    raise ValueError(name)
