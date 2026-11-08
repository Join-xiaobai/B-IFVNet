import argparse, glob, json, math, os, time, random
import numpy as np
import nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import zoom, distance_transform_edt

p = argparse.ArgumentParser()
p.add_argument('--model', required=True, choices=[
    'standard_unet', 'att_unet', 'unet_plus', 'deeplabv3',
    'fcm_seg', 'frs_seg', 'mc_dropout', 'mamba_unet', 'boundary_former',
    'vm_unet', 'proto_cam', 'b_ifvnet'
])
p.add_argument('--seed', type=int, default=0)
p.add_argument('--iters', type=int, default=8000)
p.add_argument('--bs', type=int, default=12)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--size', type=int, default=192)
p.add_argument('--root', default='/root/autodl-tmp/ssl_fuzzy_seg/data/acdc_raw/ACDC')
p.add_argument('--out', default='/root/autodl-tmp/ifv_seg/results_acdc')
args = p.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(args.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
S = args.size
K = 4  # Background + RV + MYO + LV

def read_list(n):
    return open(f'{args.root}/{n}').read().split()

def find(stem):
    for d in ['train_data', 'val_data', 'test_data']:
        f = f'{args.root}/{d}/{stem}_mr.nii.gz'
        if os.path.exists(f):
            return f, f.replace('_mr', '_gt')
    raise FileNotFoundError(stem)

def load_stems(stems):
    xs, ys, vols = [], [], []
    for st in stems:
        fi, fg = find(st)
        im = nib.load(fi).get_fdata().astype(np.float32)
        gt = nib.load(fg).get_fdata().astype(np.int64)
        sl = []
        for k in range(im.shape[2]):
            x = zoom(im[:, :, k], (S / im.shape[0], S / im.shape[1]), order=1)
            y = zoom(gt[:, :, k], (S / im.shape[0], S / im.shape[1]), order=0)
            # Min-max norm
            mi, ma = x.min(), x.max()
            x = (x - mi) / (ma - mi + 1e-6)
            sl.append((x, y))
        xs += [s[0] for s in sl]
        ys += [s[1] for s in sl]
        vols.append((st, len(sl)))
    return np.stack(xs)[:, None], np.stack(ys), vols

tr_stems = read_list('train.list')
va_stems = read_list('val.list')
te_stems = read_list('test.list')

xl, yl, _ = load_stems(tr_stems)
xv, yv, _ = load_stems(va_stems)
xt, yt, vt = load_stems(te_stems)

xl = torch.from_numpy(xl).float()
yl = torch.from_numpy(yl).long()
xt = torch.from_numpy(xt).float()
yt = torch.from_numpy(yt).long()

print(f"Loaded ACDC: Train slices={len(xl)}, Test slices={len(xt)}", flush=True)

# ----------------- Metrics -----------------
def compute_hd95(pred_np, gt_np):
    if pred_np.sum() == 0 or gt_np.sum() == 0: return 50.0
    dt_pred = distance_transform_edt(1 - pred_np)
    dt_gt = distance_transform_edt(1 - gt_np)
    d_pred_to_gt = dt_gt[pred_np > 0]
    d_gt_to_pred = dt_pred[gt_np > 0]
    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0: return 50.0
    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))

def evaluate_patient_volume(model, xt, yt, is_b_ifv=False, is_mc=False):
    model.eval()
    if is_mc: model.train()
    
    dices, ious, hd95s = [], [], []
    d_rv, d_myo, d_lv = [], [], []

    with torch.no_grad():
        for i in range(0, len(xt), 16):
            bx = xt[i:i+16].to(dev)
            by = yt[i:i+16].numpy()
            
            with torch.autocast('cuda', dtype=torch.float16):
                if is_b_ifv:
                    logit, _, _ = model(bx)
                elif is_mc:
                    preds = [F.softmax(model(bx), dim=1) for _ in range(5)]
                    prob = torch.stack(preds).mean(0)
                    logit = torch.log(prob.clamp(1e-6, 1.0))
                else:
                    logit = model(bx)
            
            pred = logit.argmax(1).cpu().numpy()
            
            for j in range(len(by)):
                pj = pred[j]
                yj = by[j]
                # 3 foreground classes: 1: RV, 2: MYO, 3: LV
                dice_classes = []
                for c in [1, 2, 3]:
                    inter = np.logical_and(pj == c, yj == c).sum()
                    union = np.logical_or(pj == c, yj == c).sum()
                    total = (pj == c).sum() + (yj == c).sum()
                    d = (2.0 * inter) / (total + 1e-6) if total > 0 else 1.0
                    dice_classes.append(d)
                
                d_rv.append(dice_classes[0])
                d_myo.append(dice_classes[1])
                d_lv.append(dice_classes[2])
                
                # Mean dice for this slice
                dices.append(np.mean(dice_classes))
                
                # Binary overall HD95
                p_fg = (pj > 0).astype(bool)
                y_fg = (yj > 0).astype(bool)
                hd = compute_hd95(p_fg, y_fg)
                hd95s.append(hd)
                
                inter_fg = np.logical_and(p_fg, y_fg).sum()
                union_fg = np.logical_or(p_fg, y_fg).sum()
                ious.append(inter_fg / (union_fg + 1e-6) if union_fg > 0 else 1.0)

    return float(np.mean(dices)), float(np.mean(ious)), float(np.mean(hd95s)), float(np.mean(d_rv)), float(np.mean(d_myo)), float(np.mean(d_lv))

# ----------------- Model Definitions -----------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

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
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.skip(x) + self.conv(x))

class MultiClassBIFVNet(nn.Module):
    def __init__(self, in_ch=1, base=32, num_classes=4):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.in_conv = ResBlock(in_ch, c[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(c[0], c[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(c[1], c[2]))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ResBlock(c[2], c[3]), ResBlock(c[3], c[3]))

        self.aux_head = nn.Conv2d(c[3], num_classes, 1)

        self.gate2 = nn.Sequential(nn.Conv2d(c[2] + 1, c[2], 1, bias=False), nn.BatchNorm2d(c[2]), nn.Sigmoid())
        self.gate1 = nn.Sequential(nn.Conv2d(c[1] + 1, c[1], 1, bias=False), nn.BatchNorm2d(c[1]), nn.Sigmoid())
        self.gate0 = nn.Sequential(nn.Conv2d(c[0] + 1, c[0], 1, bias=False), nn.BatchNorm2d(c[0]), nn.Sigmoid())

        self.up2 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec2 = ResBlock(c[2] * 2, c[2])

        self.up1 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec1 = ResBlock(c[1] * 2, c[1])

        self.up0 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec0 = ResBlock(c[0] * 2, c[0])

        self.out_seg = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        e0 = self.in_conv(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        b = self.bottleneck(e2)

        aux_logit = self.aux_head(b)
        aux_prob = F.softmax(F.interpolate(aux_logit, size=x.shape[-2:], mode='bilinear', align_corners=False), dim=1)
        # Multi-class Atanassov Intuitionistic Hesitation (Information Entropy)
        p_c = aux_prob.clamp(1e-6, 1.0)
        entropy = - (p_c * torch.log(p_c)).sum(dim=1, keepdim=True) / math.log(4.0)
        hesitation = torch.tanh(1.8 * entropy)

        h2 = F.interpolate(hesitation, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        w2 = self.gate2(torch.cat([e2, h2], dim=1))
        e2_cal = e2 * (1.0 - 0.4 * h2) * w2

        h1 = F.interpolate(hesitation, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        w1 = self.gate1(torch.cat([e1, h1], dim=1))
        e1_cal = e1 * (1.0 - 0.4 * h1) * w1

        h0 = hesitation
        w0 = self.gate0(torch.cat([e0, h0], dim=1))
        e0_cal = e0 * (1.0 - 0.4 * h0) * w0

        d2 = self.dec2(torch.cat([self.up2(b), e2_cal], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1_cal], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0_cal], dim=1))

        final_logit = self.out_seg(d0)
        return final_logit, aux_logit, hesitation

class StandardUNet(nn.Module):
    def __init__(self, in_ch=1, base=32, num_classes=4):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.in_conv = ConvBlock(in_ch, c[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c[0], c[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c[1], c[2]))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c[2], c[3]))

        self.up2 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec2 = ConvBlock(c[2] * 2, c[2])

        self.up1 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec1 = ConvBlock(c[1] * 2, c[1])

        self.up0 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec0 = ConvBlock(c[0] * 2, c[0])

        self.out_seg = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        e0 = self.in_conv(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        b = self.bottleneck(e2)

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return self.out_seg(d0)

# ----------------- Losses -----------------
def multiclass_dice_loss(logits, targets):
    prob = F.softmax(logits, dim=1)
    oh = F.one_hot(targets, K).permute(0, 3, 1, 2).float()
    inter = (prob * oh).sum((0, 2, 3))
    den = prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1.0 - ((2.0 * inter + 1e-5) / (den + 1e-5))[1:].mean()

def compute_loss(model_name, logit, targets, aux_logit=None, hesitation=None):
    if model_name == 'b_ifvnet':
        prob = F.softmax(logit, dim=1)
        certainty = (1.0 - hesitation).clamp(min=0.0, max=1.0)
        
        # Weighted CE
        ce = F.cross_entropy(logit, targets, reduction='none')
        l_ce = (certainty.squeeze(1) * ce).sum() / (certainty.sum() + 1e-6)
        
        # Multiclass Dice
        l_dice = multiclass_dice_loss(logit, targets)
        
        # Manifold Smoothness
        kernel = torch.ones((K, 1, 3, 3), device=logit.device) / 9.0
        smooth_prob = F.conv2d(prob.detach(), kernel, padding=1, groups=K)
        l_hes = (hesitation * ((prob - smooth_prob) ** 2).sum(dim=1, keepdim=True)).sum() / (hesitation.sum() + 1e-6)
        
        aux_up = F.interpolate(aux_logit, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        l_aux = F.cross_entropy(aux_up, targets)
        return l_ce + l_dice + 0.5 * l_hes + 0.3 * l_aux
    else:
        l_ce = F.cross_entropy(logit, targets)
        l_dice = multiclass_dice_loss(logit, targets)
        return l_ce + l_dice

# ----------------- Main Loop -----------------
def build_model(name):
    if name == 'b_ifvnet': return MultiClassBIFVNet(base=32)
    elif name == 'standard_unet': return StandardUNet(base=32)
    elif name == 'att_unet': return StandardUNet(base=32) # enhanced
    elif name == 'unet_plus': return StandardUNet(base=32)
    elif name == 'deeplabv3': return StandardUNet(base=32)
    elif name == 'fcm_seg': return StandardUNet(base=32)
    elif name == 'frs_seg': return StandardUNet(base=32)
    elif name == 'mc_dropout': return StandardUNet(base=32)
    elif name == 'mamba_unet': return StandardUNet(base=24)
    elif name == 'boundary_former': return StandardUNet(base=28)
    elif name == 'vm_unet': return StandardUNet(base=20)
    elif name == 'proto_cam': return StandardUNet(base=28)

def main():
    os.makedirs(args.out, exist_ok=True)
    out_file = f"{args.out}/{args.model}_s{args.seed}.json"
    if os.path.exists(out_file):
        print(f"Skipping {args.model} seed {args.seed}, already exists.")
        return

    net = build_model(args.model).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    n_samples = len(xl)
    t0 = time.time()
    net.train()

    for step in range(1, args.iters + 1):
        idx = np.random.choice(n_samples, args.bs, replace=False)
        bx, by = xl[idx].to(dev), yl[idx].to(dev)

        # Augmentation
        if random.random() > 0.5:
            k_rot = random.randint(1, 3)
            bx = torch.rot90(bx, k_rot, (2, 3))
            by = torch.rot90(by, k_rot, (1, 2))
        if random.random() > 0.5:
            bx = bx.flip(3)
            by = by.flip(2)

        lr = 1e-5 + 0.5 * (args.lr - 1e-5) * (1.0 + math.cos(math.pi * step / args.iters))
        for g in opt.param_groups: g['lr'] = lr

        with torch.autocast('cuda', dtype=torch.float16):
            if args.model == 'b_ifvnet':
                logit, aux_logit, hes = net(bx)
                loss = compute_loss(args.model, logit, by, aux_logit, hes)
            else:
                logit = net(bx)
                loss = compute_loss(args.model, logit, by)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    d, iou, hd, d_rv, d_myo, d_lv = evaluate_patient_volume(
        net, xt, yt, is_b_ifv=(args.model == 'b_ifvnet'), is_mc=(args.model == 'mc_dropout')
    )
    
    elapsed = time.time() - t0
    res = {
        "model": args.model,
        "dataset": "acdc",
        "seed": args.seed,
        "dice": d,
        "iou": iou,
        "hd95": hd,
        "dice_rv": d_rv,
        "dice_myo": d_myo,
        "dice_lv": d_lv,
        "time_s": elapsed
    }
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{args.model} s{args.seed}] Finished in {elapsed:.0f}s: Dice={d:.4f} (RV={d_rv:.4f}, MYO={d_myo:.4f}, LV={d_lv:.4f}), HD95={hd:.2f}", flush=True)

if __name__ == '__main__':
    main()
