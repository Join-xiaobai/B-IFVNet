import os, sys, time, json, math, random, glob
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as TF

# ----------------- Seed & Device -----------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_hd95(pred_np, gt_np):
    if pred_np.sum() == 0 or gt_np.sum() == 0: return 50.0
    dt_pred = distance_transform_edt(1 - pred_np)
    dt_gt = distance_transform_edt(1 - gt_np)
    d_pred_to_gt = dt_gt[pred_np > 0]
    d_gt_to_pred = dt_pred[gt_np > 0]
    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0: return 50.0
    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))

# ----------------- Kvasir Dataset -----------------
class KvasirDataset(Dataset):
    def __init__(self, img_paths, mask_paths, size=256, is_train=True):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = size
        self.is_train = is_train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.is_train:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

        img = TF.resize(img, (self.size, self.size))
        mask = TF.resize(mask, (self.size, self.size), interpolation=Image.NEAREST)

        img_t = TF.to_tensor(img)  # 3 x H x W in [0, 1]
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_np = np.array(mask, dtype=np.float32)
        mask_t = torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t

# ----------------- Optical Multi-Scale Boundary Operator -----------------
def compute_optical_boundary_response(img_tensor):
    gx = torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1])
    gy = torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :])
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).mean(dim=1, keepdim=True)
    norm_grad = grad / (grad.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return norm_grad

# ----------------- Residual Block with Squeeze-and-Excitation -----------------
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(4, ch // r), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, ch // r), ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.skip(x) + self.se(self.conv(x)))

# ----------------- Intuitionistic Hesitation-Calibrated Skip Gate -----------------
class HesitationGate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(ch + 1, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.Sigmoid()
        )

    def forward(self, skip_feat, hesitation):
        hes_down = F.interpolate(hesitation, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
        weight = self.gate_conv(torch.cat([skip_feat, hes_down], dim=1))
        return skip_feat * (1.0 - 0.4 * hes_down) * weight

# ----------------- B-IFVNet Optical Architecture -----------------
class OpticalBIFVNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        # 256
        self.in_conv = ResBlock(in_ch, c[0])
        # 128
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(c[0], c[1]))
        # 64
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(c[1], c[2]))
        # 32
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(c[2], c[3]),
            ResBlock(c[3], c[3])
        )

        self.aux_head = nn.Conv2d(c[3], 1, 1)

        self.gate2 = HesitationGate(c[2])
        self.gate1 = HesitationGate(c[1])
        self.gate0 = HesitationGate(c[0])

        self.up2 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec2 = ResBlock(c[2] * 2, c[2])

        self.up1 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec1 = ResBlock(c[1] * 2, c[1])

        self.up0 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec0 = ResBlock(c[0] * 2, c[0])

        self.out_seg = nn.Conv2d(c[0], 1, 1)

    def forward(self, x, boundary_resp=None):
        e0 = self.in_conv(x)       # 256, c[0]
        e1 = self.down1(e0)        # 128, c[1]
        e2 = self.down2(e1)        # 64,  c[2]
        b = self.bottleneck(e2)    # 32,  c[3]

        aux_logit = self.aux_head(b)
        aux_prob = torch.sigmoid(F.interpolate(aux_logit, size=x.shape[-2:], mode='bilinear', align_corners=False))
        p_c = aux_prob.clamp(1e-6, 1.0 - 1e-6)
        entropy = - (p_c * torch.log2(p_c) + (1.0 - p_c) * torch.log2(1.0 - p_c))
        
        if boundary_resp is not None:
            hesitation = torch.tanh(1.5 * entropy + 0.6 * boundary_resp)
        else:
            hesitation = torch.tanh(1.5 * entropy)

        d2 = self.dec2(torch.cat([self.up2(b), self.gate2(e2, hesitation)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.gate1(e1, hesitation)], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), self.gate0(e0, hesitation)], dim=1))

        final_logit = self.out_seg(d0)
        return final_logit, aux_logit, hesitation

# ----------------- Bi-Directional IFV Loss -----------------
class BiDirectionalOpticalIFVLoss(nn.Module):
    def __init__(self, lambda_hes=0.6, eps=1e-6):
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

# ----------------- Evaluation -----------------
def evaluate_kvasir(net, val_loader):
    net.eval()
    dices, ious, hd95s = [], [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logit, _, _ = net(bx)
            pred = (torch.sigmoid(logit) > 0.5).float()
            
            p_np = pred.squeeze(1).cpu().numpy().astype(bool)
            t_np = by.squeeze(1).cpu().numpy().astype(bool)

            for i in range(len(p_np)):
                pi, ti = p_np[i], t_np[i]
                inter = np.logical_and(pi, ti).sum()
                union = np.logical_or(pi, ti).sum()
                total = pi.sum() + ti.sum()
                d = (2.0 * inter) / (total + 1e-6) if total > 0 else 1.0
                u = inter / (union + 1e-6) if union > 0 else 1.0
                hd = compute_hd95(pi, ti)
                dices.append(d)
                ious.append(u)
                hd95s.append(hd)
    return float(np.mean(dices)), float(np.mean(ious)), float(np.mean(hd95s))

# ----------------- Training Loop -----------------
def run_experiment(args):
    img_files = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")) + glob.glob(os.path.join(args.img_dir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*.jpg")) + glob.glob(os.path.join(args.mask_dir, "*.png")))
    assert len(img_files) == len(mask_files) and len(img_files) > 0, "No image/mask pairs found"
    
    n = len(img_files)
    idx = list(range(n))
    seed_everything(args.rep * 1000 + 42)
    random.shuffle(idx)

    k = args.folds
    fold_size = n // k
    results = []

    os.makedirs(args.out, exist_ok=True)

    for fold in range(k):
        t0 = time.time()
        val_idx = idx[fold * fold_size : (fold + 1) * fold_size]
        tr_idx = [i for i in idx if i not in val_idx]

        tr_imgs = [img_files[i] for i in tr_idx]
        tr_masks = [mask_files[i] for i in tr_idx]
        val_imgs = [img_files[i] for i in val_idx]
        val_masks = [mask_files[i] for i in val_idx]

        tr_ds = KvasirDataset(tr_imgs, tr_masks, size=args.size, is_train=True)
        val_ds = KvasirDataset(val_imgs, val_masks, size=args.size, is_train=False)

        tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

        model = OpticalBIFVNet(in_ch=3, base=32).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        loss_fn = BiDirectionalOpticalIFVLoss(lambda_hes=0.6)

        total_steps = args.iters
        step = 0
        model.train()
        
        while step < total_steps:
            for bx, by in tr_loader:
                step += 1
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                
                lr = 1e-5 + 0.5 * (args.lr - 1e-5) * (1.0 + math.cos(math.pi * step / total_steps))
                for g in opt.param_groups: g['lr'] = lr

                with torch.autocast("cuda", dtype=torch.float16):
                    bnd_resp = compute_optical_boundary_response(bx)
                    logit, aux_logit, hes = model(bx, boundary_resp=bnd_resp)
                    loss = loss_fn(logit, aux_logit, by, hes)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                if step >= total_steps:
                    break

        d, iou, hd = evaluate_kvasir(model, val_loader)
        results.append({"dice": d, "iou": iou, "hd95": hd})
        print(f"[Optical B-IFVNet Kvasir rep{args.rep}] fold {fold}: Dice={d:.4f}, IoU={iou:.4f}, HD95={hd:.2f} ({time.time()-t0:.0f}s)", flush=True)

    summary = {
        "model": "b_ifvnet",
        "dataset": "kvasir",
        "rep": args.rep,
        "folds": results,
        "mean_dice": float(np.mean([f["dice"] for f in results])),
        "mean_iou": float(np.mean([f["iou"] for f in results])),
        "mean_hd95": float(np.mean([f["hd95"] for f in results]))
    }
    out_file = os.path.join(args.out, f"b_ifvnet_rep{args.rep}.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_file}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default="/root/autodl-tmp/ssl_fuzzy_seg/data/kvasir/images")
    p.add_argument("--mask_dir", default="/root/autodl-tmp/ssl_fuzzy_seg/data/kvasir/masks")
    p.add_argument("--out", default="/root/autodl-tmp/ifv_seg/results_kvasir")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--lr", type=float, default=7e-4)
    args = p.parse_args()
    run_experiment(args)
