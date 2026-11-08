import json
import numpy as np
from scipy import stats

def load_cv_results(path):
    with open(path) as f:
        data = json.load(f)
    return data

# Baseline paths (from cv_seg)
sup0 = json.load(open("/root/autodl-tmp/cv_seg/results/sup_rep0.json"))["fold_dice"]
sup1 = json.load(open("/root/autodl-tmp/cv_seg/results/sup_rep1.json"))["fold_dice"]

fcm0 = json.load(open("/root/autodl-tmp/cv_seg/results/fcm_rep0.json"))["fold_dice"]
fcm1 = json.load(open("/root/autodl-tmp/cv_seg/results/fcm_rep1.json"))["fold_dice"]

frs0 = json.load(open("/root/autodl-tmp/cv_seg/results/frs_rep0.json"))["fold_dice"]
frs1 = json.load(open("/root/autodl-tmp/cv_seg/results/frs_rep1.json"))["fold_dice"]

# IFVNet results
ifv_u0 = [x["dice"] for x in json.load(open("/root/autodl-tmp/ifv_seg/results/ifvnet_unet16_rep0.json"))["folds"]]
ifv_u1 = [x["dice"] for x in json.load(open("/root/autodl-tmp/ifv_seg/results/ifvnet_unet16_rep1.json"))["folds"]]

ifv_res0_d = [x["dice"] for x in json.load(open("/root/autodl-tmp/ifv_seg/results/ifvnet_res_unet_rep0.json"))["folds"]]
ifv_res1_d = [x["dice"] for x in json.load(open("/root/autodl-tmp/ifv_seg/results/ifvnet_res_unet_rep1.json"))["folds"]]

ifv_res0_iou = [x["iou"] for x in json.load(open("/root/autodl-tmp/ifv_seg/results/ifvnet_res_unet_rep0.json"))["folds"]]
ifv_res1_iou = [x["iou"] for x in json.load(open("/root/autodl-tmp/ifv_seg/results/ifvnet_res_unet_rep1.json"))["folds"]]

ifv_res0_hd = [x["hd95"] for x in json.load(open("/root/autodl-tmp/ifv_seg/results/ifvnet_res_unet_rep0.json"))["folds"]]
ifv_res1_hd = [x["hd95"] for x in json.load(open("/root/autodl-tmp/ifv_seg/results/ifvnet_res_unet_rep1.json"))["folds"]]

all_sup = sup0 + sup1
all_fcm = fcm0 + fcm1
all_frs = frs0 + frs1
all_ifv_res = ifv_res0_d + ifv_res1_d
all_ifv_iou = ifv_res0_iou + ifv_res1_iou
all_ifv_hd = ifv_res0_hd + ifv_res1_hd

print("=== 10-Fold Overall Comparison (Dice) ===")
print(f"Supervised: {np.mean(all_sup):.4f} ± {np.std(all_sup):.4f}")
print(f"FCM:        {np.mean(all_fcm):.4f} ± {np.std(all_fcm):.4f}")
print(f"FRS:        {np.mean(all_frs):.4f} ± {np.std(all_frs):.4f}")
print(f"SA-IFVNet (ResUNet): {np.mean(all_ifv_res):.4f} ± {np.std(all_ifv_res):.4f}")
print(f"  IoU:      {np.mean(all_ifv_iou):.4f} ± {np.std(all_ifv_iou):.4f}")
print(f"  HD95:     {np.mean(all_ifv_hd):.2f} ± {np.std(all_ifv_hd):.2f}")

# Wilcoxon signed-rank test against all baselines
w_sup = stats.wilcoxon(all_ifv_res, all_sup, alternative='greater')
w_fcm = stats.wilcoxon(all_ifv_res, all_fcm, alternative='greater')
w_frs = stats.wilcoxon(all_ifv_res, all_frs, alternative='greater')

print("\n=== Wilcoxon Signed-Rank Tests (p-values) ===")
print(f"SA-IFVNet vs Supervised: stat={w_sup.statistic:.1f}, p={w_sup.pvalue:.4e} (p < 0.05: {w_sup.pvalue < 0.05})")
print(f"SA-IFVNet vs FCM:        stat={w_fcm.statistic:.1f}, p={w_fcm.pvalue:.4e} (p < 0.05: {w_fcm.pvalue < 0.05})")
print(f"SA-IFVNet vs FRS:        stat={w_frs.statistic:.1f}, p={w_frs.pvalue:.4e} (p < 0.05: {w_frs.pvalue < 0.05})")
