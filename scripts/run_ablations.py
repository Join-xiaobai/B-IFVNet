import argparse, glob, json, os, time
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--mode", required=True, choices=["m1_no_acoustic", "m2_no_manifold", "sweep_lambda", "sweep_gamma"])
p.add_argument("--data", default="/root/autodl-tmp/ssl_fuzzy_seg/data/busi_raw")
p.add_argument("--out", default="/root/autodl-tmp/ifv_seg/results_ablation_fine")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--rep", type=int, default=0)
p.add_argument("--iters", type=int, default=8000)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--size", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
S = args.size

def load_busi(root, size):
    imgs, masks = [], []
    for cls in ["benign", "malignant"]:
        files = sorted([f for f in glob.glob(f"{root}/{cls}/*.png") if "mask" not in f])
        for img_f in files:
            mask_f = img_f.replace(".png", "_mask.png")
            if not os.path.exists(mask_f): continue
            img = np.array(Image.open(img_f).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
            mask = (np.array(Image.open(mask_f).convert("L").resize((size, size), Image.NEAREST)) > 127).astype(np.float32)
            imgs.append(img)
            masks.append(mask)
    return torch.from_numpy(np.stack(imgs)).unsqueeze(1), torch.from_numpy(np.stack(masks)).unsqueeze(1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // 4, 1), nn.ReLU(True),
            nn.Conv2d(channels // 4, channels, 1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        res = self.conv(x)
        w = self.se(res)
        return self.relu(x + res * w)

class HesitationCalibratedGate(nn.Module):
    def __init__(self, in_ch, gamma=0.5):
        super().__init__()
        self.gamma = gamma
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch + 1, in_ch, 1, bias=False), nn.BatchNorm2d(in_ch), nn.Sigmoid()
        )

    def forward(self, skip_feat, hesitation_map):
        hes_down = F.interpolate(hesitation_map, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
        weight = self.gate_conv(torch.cat([skip_feat, hes_down], dim=1))
        return skip_feat * (1.0 - self.gamma * hes_down) * weight

class B_IFVNet_Ablation(nn.Module):
    def __init__(self, in_ch=1, base=32, use_acoustic=True, gamma=0.5):
        super().__init__()
        self.use_acoustic = use_acoustic
        c = [base, base * 2, base * 4, base * 8]
        self.in_conv = nn.Sequential(nn.Conv2d(in_ch, c[0], 3, padding=1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(True))
        self.enc1 = ResidualBlock(c[0])
        self.down1 = nn.Conv2d(c[0], c[1], 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(c[1])
        self.down2 = nn.Conv2d(c[1], c[2], 3, stride=2, padding=1)
        self.enc3 = ResidualBlock(c[2])
        self.down3 = nn.Conv2d(c[2], c[3], 3, stride=2, padding=1)
        self.bottleneck = ResidualBlock(c[3])

        self.gate3 = HesitationCalibratedGate(c[2], gamma=gamma)
        self.gate2 = HesitationCalibratedGate(c[1], gamma=gamma)
        self.gate1 = HesitationCalibratedGate(c[0], gamma=gamma)
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

        if self.use_acoustic and acoustic_grad is not None:
            hesitation = torch.tanh(1.5 * entropy + 0.5 * acoustic_grad)
        else:
            hesitation = torch.tanh(1.5 * entropy)

        e3_cal = self.gate3(e3, hesitation)
        d3 = self.dec3(torch.cat([self.up3(b), e3_cal], dim=1))
        e2_cal = self.gate2(e2, hesitation)
        d2 = self.dec2(torch.cat([self.up2(d3), e2_cal], dim=1))
        e1_cal = self.gate1(e1, hesitation)
        d1 = self.dec1(torch.cat([self.up1(d2), e1_cal], dim=1))

        return self.out_seg(d1), aux_logit, hesitation

class Loss_Ablation(nn.Module):
    def __init__(self, lambda_hes=0.5, use_manifold=True):
        super().__init__()
        self.lambda_hes = lambda_hes
        self.use_manifold = use_manifold

    def forward(self, final_logit, aux_logit, targets, hesitation):
        prob = torch.sigmoid(final_logit)
        certainty = (1.0 - hesitation).clamp(min=0.0, max=1.0)
        bce = F.binary_cross_entropy_with_logits(final_logit, targets, reduction='none')
        l_bce = (certainty * bce).sum() / (certainty.sum() + 1e-6)

        inter = (prob * targets * certainty).sum(dim=(-2, -1))
        union = (prob * certainty).sum(dim=(-2, -1)) + (targets * certainty).sum(dim=(-2, -1))
        l_dice = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()

        if self.use_manifold and self.lambda_hes > 0:
            kernel = torch.ones((1, 1, 3, 3), device=final_logit.device) / 9.0
            smooth_prob = F.conv2d(prob.detach(), kernel, padding=1)
            l_hes = (hesitation * (prob - smooth_prob) ** 2).sum() / (hesitation.sum() + 1e-6)
        else:
            l_hes = torch.tensor(0.0, device=final_logit.device)

        aux_up = F.interpolate(aux_logit, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        l_aux = F.binary_cross_entropy_with_logits(aux_up, targets)
        return l_bce + l_dice + self.lambda_hes * l_hes + 0.3 * l_aux

def compute_hd95(pred_np, gt_np):
    if pred_np.sum() == 0 or gt_np.sum() == 0: return 50.0
    dt_pred = distance_transform_edt(1 - pred_np)
    dt_gt = distance_transform_edt(1 - gt_np)
    d_pred_to_gt = dt_gt[pred_np > 0]
    d_gt_to_pred = dt_pred[gt_np > 0]
    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0: return 50.0
    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))

@torch.no_grad()
def evaluate_metrics(net, x, y):
    net.eval()
    dices, ious, hd95s = [], [], []
    for i in range(0, len(x), 32):
        bx, by = x[i:i + 32], y[i:i + 32]
        with torch.autocast("cuda", dtype=torch.float16):
            logit, _, _ = net(bx)
        prob = torch.sigmoid(logit.float())
        pred = (prob > 0.5).float()
        inter = (pred * by).sum((1, 2, 3))
        union = pred.sum((1, 2, 3)) + by.sum((1, 2, 3))
        dices.extend(((2 * inter + 1e-6) / (union + 1e-6)).cpu().tolist())
        ious.extend(((inter + 1e-6) / (union - inter + 1e-6)).cpu().tolist())
        for j in range(len(bx)):
            p_np = pred[j, 0].cpu().numpy().astype(bool)
            g_np = by[j, 0].cpu().numpy().astype(bool)
            hd95s.append(compute_hd95(p_np, g_np))
    net.train()
    return float(np.mean(dices)), float(np.mean(ious)), float(np.mean(hd95s))

def compute_acoustic_gradient(img):
    gx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    gy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    local_var = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return local_var / (local_var.amax(dim=(-2, -1), keepdim=True) + 1e-6)

def augment(x, y):
    if np.random.rand() < 0.5:
        k = np.random.randint(4)
        x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
    if np.random.rand() < 0.5:
        x, y = x.flip(3), y.flip(3)
    return x, y

def main():
    os.makedirs(args.out, exist_ok=True)
    X, Y = load_busi(args.data, S)
    n = len(X)
    order = np.random.RandomState(1000 + args.rep).permutation(n)
    fold_ids = np.array_split(order, args.folds)
    fold_results = []

    use_acoustic = (args.mode != "m1_no_acoustic")
    use_manifold = (args.mode != "m2_no_manifold")
    lambda_hes = 0.5
    gamma = 0.5

    loss_fn = Loss_Ablation(lambda_hes=lambda_hes, use_manifold=use_manifold)

    for fold in range(args.folds):
        te_idx = fold_ids[fold]
        tr_idx = np.concatenate([fold_ids[k] for k in range(args.folds) if k != fold])
        torch.manual_seed(fold + 100 * args.rep)
        np.random.seed(fold + 100 * args.rep)
        x_tr, y_tr = X[tr_idx].to(dev), Y[tr_idx].to(dev)
        x_te, y_te = X[te_idx].to(dev), Y[te_idx].to(dev)

        model = B_IFVNet_Ablation(in_ch=1, base=32, use_acoustic=use_acoustic, gamma=gamma).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        t0 = time.time()

        for it in range(1, args.iters + 1):
            for g in opt.param_groups:
                g["lr"] = args.lr * (1 - it / args.iters) ** 0.9
            bi = torch.randint(0, len(x_tr), (args.bs,), device=dev)
            xb, yb = augment(x_tr[bi], y_tr[bi])
            grad_b = compute_acoustic_gradient(xb) if use_acoustic else None

            with torch.autocast("cuda", dtype=torch.float16):
                final_logit, aux_logit, hesitation = model(xb, acoustic_grad=grad_b)
                loss = loss_fn(final_logit, aux_logit, yb, hesitation)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        d, iou, hd = evaluate_metrics(model, x_te, y_te)
        fold_results.append({"dice": d, "iou": iou, "hd95": hd})
        print(f"[{args.mode} rep{args.rep}] fold {fold}: Dice={d:.4f}, IoU={iou:.4f}, HD95={hd:.2f} ({time.time()-t0:.0f}s)", flush=True)

    summary = {
        "mode": args.mode,
        "rep": args.rep,
        "folds": fold_results,
        "mean_dice": float(np.mean([f["dice"] for f in fold_results])),
        "mean_iou": float(np.mean([f["iou"] for f in fold_results])),
        "mean_hd95": float(np.mean([f["hd95"] for f in fold_results]))
    }
    json.dump(summary, open(f"{args.out}/{args.mode}_rep{args.rep}.json", "w"), indent=2)
    print(f"FINAL {args.mode} rep{args.rep}: Mean Dice = {summary['mean_dice']:.4f}, IoU = {summary['mean_iou']:.4f}, HD95 = {summary['mean_hd95']:.2f}")

if __name__ == "__main__":
    main()
