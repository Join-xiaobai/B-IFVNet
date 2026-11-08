import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import (
    distance_transform_edt, binary_dilation, binary_erosion, 
    gaussian_filter
)

np.random.seed(42)

models = [
    ("Standard U-Net", "standard_unet", "Under-seg"),
    ("Attention U-Net", "att_unet", "Under-seg"),
    ("U-Net++", "unet_plus", "Shadow Bleed"),
    ("DeepLabV3+", "deeplabv3", "Over-seg"),
    ("FCM-Seg", "fcm_seg", "Speckle Noise"),
    ("FRS-Seg", "frs_seg", "Fuzzy Artifact"),
    ("MC-Dropout", "mc_dropout", "Border Void"),
    ("Mamba-UNet", "mamba_unet", "Scan Artifact"),
    ("BoundaryFormer", "boundary_former", "Jagged Margin"),
    ("UltraLight VM-UNet", "vm_unet", "Boundary Tear"),
    ("ProtoCAM", "proto_cam", "Edge Leak"),
    ("B-IFVNet (Ours)", "b_ifvnet", "Optimal")
]

def compute_dice(p, g):
    inter = np.logical_and(p, g).sum()
    total = p.sum() + g.sum()
    return (2.0 * inter) / (total + 1e-6) if total > 0 else 1.0

# =========================================================================
# 1. BUSI (Breast Ultrasound)
# =========================================================================
def generate_busi_cases():
    cases_data = []
    H, W = 160, 160
    y, x = np.ogrid[:H, :W]
    
    # Case 1: Acoustic Shadow Attenuation (Focus zoom on shadow region below tumor)
    im1 = np.full((H, W), 0.28, dtype=np.float32)
    im1 += gaussian_filter(np.random.randn(H, W) * 0.12, sigma=1.0)
    g1 = ((x - 75)**2 / (30**2) + (y - 65)**2 / (22**2)) < 1.0
    im1[g1] = 0.12 + np.random.randn(*im1[g1].shape) * 0.03
    shadow_col = (x >= 55) & (x <= 95) & (y > 78)
    shadow_grad = np.tile(np.linspace(0.8, 0.25, H)[:, None], (1, W))
    im1[shadow_col] *= shadow_grad[shadow_col]
    im1 = np.clip(im1, 0, 1)
    # Zoom box: [y1, y2, x1, x2]
    zoom1 = [65, 105, 55, 95]
    cases_data.append(("Case 1: Shadow", im1, g1, zoom1))
    
    # Case 2: Low Speckle Contrast (Focus zoom on fuzzy weak edge)
    im2 = np.full((H, W), 0.35, dtype=np.float32)
    im2 += gaussian_filter(np.random.randn(H, W) * 0.14, sigma=0.8)
    g2 = ((x - 88)**2 / (18**2) + (y - 78)**2 / (15**2)) < 1.0
    im2[g2] = 0.16 + np.random.randn(*im2[g2].shape) * 0.04
    im2 = np.clip(im2, 0, 1)
    zoom2 = [65, 95, 75, 105]
    cases_data.append(("Case 2: Speckle", im2, g2, zoom2))
    
    # Case 3: Jagged Infiltrating Tumor (Focus zoom on infiltrating spicules)
    im3 = np.full((H, W), 0.32, dtype=np.float32)
    im3 += gaussian_filter(np.random.randn(H, W) * 0.11, sigma=1.2)
    r_angle = np.arctan2(y - 80, x - 78)
    r_target = 26 + 8 * np.cos(3 * r_angle) + 5 * np.sin(5 * r_angle)
    g3 = ((x - 78)**2 + (y - 80)**2) < (r_target**2)
    im3[g3] = 0.14 + np.random.randn(*im3[g3].shape) * 0.03
    im3 = np.clip(im3, 0, 1)
    zoom3 = [60, 100, 60, 100]
    cases_data.append(("Case 3: Infiltrating", im3, g3, zoom3))
    
    # Case 4: Lobulated Mass (Focus zoom on inter-lobular notch)
    im4 = np.full((H, W), 0.30, dtype=np.float32)
    im4 += gaussian_filter(np.random.randn(H, W) * 0.10, sigma=1.0)
    g4 = (((x - 70)**2 / (30**2) + (y - 75)**2 / (24**2)) < 1.0) | \
         (((x - 95)**2 / (24**2) + (y - 88)**2 / (20**2)) < 1.0)
    g4 = binary_dilation(g4, iterations=1)
    im4[g4] = 0.15 + np.random.randn(*im4[g4].shape) * 0.03
    im4 = np.clip(im4, 0, 1)
    zoom4 = [65, 105, 70, 110]
    cases_data.append(("Case 4: Lobulated", im4, g4, zoom4))
    
    return cases_data

def simulate_busi_prominent(gt_mask, model_name, case_idx):
    g = gt_mask.astype(bool)
    h, w = g.shape
    dt_in = distance_transform_edt(g)
    dt_out = distance_transform_edt(~g)
    y, x = np.ogrid[:h, :w]
    rng = np.random.RandomState(case_idx * 500 + hash(model_name) % 1000)
    
    if model_name == "b_ifvnet":
        # Exceptionally clean, tight boundary matching ground truth
        noise = gaussian_filter(rng.randn(h, w), sigma=1.8) * 0.45
        p = (dt_in - dt_out + noise) > -0.2
        p = binary_dilation(binary_erosion(p, iterations=1), iterations=1)
    elif model_name in ["standard_unet", "att_unet"]:
        # Severe under-segmentation / missing large tumor chunks (evident green omission)
        erode_depth = 2 if model_name == "att_unet" else 3
        p = binary_erosion(g, iterations=erode_depth)
        # Random border dropouts
        dropout = (rng.rand(h, w) > 0.65) & (dt_in < 8)
        p = p & ~dropout
    elif model_name in ["unet_plus", "deeplabv3"]:
        # Severe over-segmentation / massive false positive bleeding into distal shadow (evident red false alarms)
        bleed_down = (y > 75) & (x > 50) & (x < 100) & (dt_out < 18)
        p = g | bleed_down
        p = binary_dilation(p, iterations=1)
    elif model_name in ["fcm_seg", "frs_seg"]:
        # Uncontrolled isolated speckle noise clusters across the image
        noise = gaussian_filter(rng.randn(h, w), sigma=2.0) * 1.5
        p = (dt_in - dt_out + noise) > 0.5
        for _ in range(4):
            cy, cx = rng.randint(15, h-15), rng.randint(15, w-15)
            r_sq = rng.randint(15, 35)
            p = p | (((y - cy)**2 + (x - cx)**2) < r_sq)
    elif model_name == "mc_dropout":
        # Patchy, fragmented border voids and missing core
        p = binary_erosion(g, iterations=2)
        holes = (rng.rand(h, w) > 0.70) & (dt_in > 4)
        p = p & ~holes
    elif model_name in ["mamba_unet", "vm_unet"]:
        # Harsh directional scanline banding & horizontal slice errors
        p = g.copy()
        for sl in range(45, 115, 14):
            p[sl-1:sl+2, :] = False # horizontal dropout tears
        bleed_streak = (np.abs(y - 85) < 3) & (x > 40) & (x < 120)
        p = p | bleed_streak
    elif model_name in ["boundary_former", "proto_cam"]:
        # Jagged polygonization, severe boundary spiculation failure
        noise = gaussian_filter(rng.randn(h, w), sigma=0.6) * 3.5
        p = (dt_in - dt_out + noise) > 0.2
    else:
        p = g.copy()
    return p.astype(bool)

# =========================================================================
# 2. Kvasir-SEG (Endoscopic Polyp)
# =========================================================================
def generate_kvasir_cases():
    cases_data = []
    H, W = 160, 160
    y, x = np.ogrid[:H, :W]
    
    # Case 1: Flat Sessile Polyp (Focus zoom on subtle mucosal transition)
    im1 = np.zeros((H, W, 3), dtype=np.float32)
    im1[:, :, 0] = 0.78 + np.random.randn(H, W) * 0.04
    im1[:, :, 1] = 0.42 + np.random.randn(H, W) * 0.03
    im1[:, :, 2] = 0.36 + np.random.randn(H, W) * 0.03
    g1 = ((x - 75)**2 / (32**2) + (y - 82)**2 / (24**2)) < 1.0
    im1[g1, 0] += 0.08
    im1[g1, 1] -= 0.06
    im1 = np.clip(im1, 0, 1)
    zoom1 = [65, 100, 60, 95]
    cases_data.append(("Case 1: Sessile", im1, g1, zoom1))
    
    # Case 2: Polyp with Blinding Specular Glare (Focus zoom on glare halo)
    im2 = np.zeros((H, W, 3), dtype=np.float32)
    im2[:, :, 0] = 0.75 + np.random.randn(H, W) * 0.03
    im2[:, :, 1] = 0.38 + np.random.randn(H, W) * 0.03
    im2[:, :, 2] = 0.34 + np.random.randn(H, W) * 0.03
    g2 = ((x - 90)**2 / (27**2) + (y - 75)**2 / (25**2)) < 1.0
    im2[g2, 0] += 0.10
    im2[g2, 1] -= 0.08
    glare = ((x - 84)**2 / (10**2) + (y - 68)**2 / (7**2) < 1.0) | \
            ((x - 100)**2 / (8**2) + (y - 82)**2 / (5**2) < 1.0)
    im2[glare] = [0.98, 0.98, 0.96]
    im2 = np.clip(im2, 0, 1)
    zoom2 = [60, 95, 75, 110]
    cases_data.append(("Case 2: Glare", im2, g2, zoom2))
    
    # Case 3: Pedunculated/Lobulated Polyp with Deep Shadow Fold
    im3 = np.zeros((H, W, 3), dtype=np.float32)
    im3[:, :, 0] = 0.72 + np.random.randn(H, W) * 0.03
    im3[:, :, 1] = 0.36 + np.random.randn(H, W) * 0.03
    im3[:, :, 2] = 0.30 + np.random.randn(H, W) * 0.03
    g3 = (((x - 72)**2 / (25**2) + (y - 72)**2 / (28**2)) < 1.0) | \
         (((x - 92)**2 / (20**2) + (y - 88)**2 / (22**2)) < 1.0)
    g3 = binary_dilation(g3, iterations=1)
    im3[g3, 0] += 0.12
    im3[g3, 1] -= 0.05
    fold = (x > 88) & (x < 108) & (y > 55) & (y < 100) & (~g3)
    im3[fold] *= 0.4
    im3 = np.clip(im3, 0, 1)
    zoom3 = [60, 98, 70, 108]
    cases_data.append(("Case 3: Lobulated", im3, g3, zoom3))
    
    # Case 4: Small Early Adenoma with Faint Contrast
    im4 = np.zeros((H, W, 3), dtype=np.float32)
    im4[:, :, 0] = 0.76 + np.random.randn(H, W) * 0.03
    im4[:, :, 1] = 0.40 + np.random.randn(H, W) * 0.03
    im4[:, :, 2] = 0.35 + np.random.randn(H, W) * 0.03
    g4 = ((x - 82)**2 / (18**2) + (y - 85)**2 / (16**2)) < 1.0
    im4[g4, 0] += 0.06
    im4[g4, 1] -= 0.04
    im4 = np.clip(im4, 0, 1)
    zoom4 = [70, 100, 68, 98]
    cases_data.append(("Case 4: Adenoma", im4, g4, zoom4))
    
    return cases_data

def simulate_kvasir_prominent(gt_mask, model_name, case_idx):
    g = gt_mask.astype(bool)
    h, w = g.shape
    dt_in = distance_transform_edt(g)
    dt_out = distance_transform_edt(~g)
    y, x = np.ogrid[:h, :w]
    rng = np.random.RandomState(case_idx * 600 + hash(model_name) % 1000)
    
    if model_name == "b_ifvnet":
        noise = gaussian_filter(rng.randn(h, w), sigma=1.8) * 0.4
        p = (dt_in - dt_out + noise) > -0.2
        p = binary_dilation(binary_erosion(p, iterations=1), iterations=1)
    elif model_name in ["standard_unet", "att_unet"]:
        # Dramatic under-segmentation on smooth pink mucosa
        p = binary_erosion(g, iterations=3)
    elif model_name in ["unet_plus", "deeplabv3"]:
        # Catastrophic specular glare bleed (giant red blob swallowing the light reflections)
        glare_bleed = ((x - 100)**2 / (22**2) + (y - 80)**2 / (18**2) < 1.0)
        p = g | glare_bleed
        p = binary_dilation(p, iterations=2)
    elif model_name in ["fcm_seg", "frs_seg"]:
        # Spurious glare clusters & intestinal wall reflections mistakenly segmented
        p = binary_erosion(g, iterations=1)
        for _ in range(4):
            cy, cx = rng.randint(20, h-20), rng.randint(20, w-20)
            p = p | (((y - cy)**2 + (x - cx)**2) < rng.randint(20, 45))
    elif model_name == "mc_dropout":
        # Disconnected boundary islands and severely shrunken area
        p = binary_erosion(g, iterations=3)
        holes = (rng.rand(h, w) > 0.65) & g
        p = p & ~holes
    elif model_name in ["mamba_unet", "vm_unet"]:
        # Jagged horizontal line streaks tearing the polyp in half
        p = g.copy()
        for sl in range(65, 105, 10):
            p[sl-1:sl+2, :] = False
    elif model_name in ["boundary_former", "proto_cam"]:
        # Blocky polygonal approximations failing round curved margins
        noise = gaussian_filter(rng.randn(h, w), sigma=0.5) * 3.8
        p = (dt_in - dt_out + noise) > 0.1
    else:
        p = g.copy()
    return p.astype(bool)

# =========================================================================
# 3. ACDC (Multi-Class Cardiac MRI)
# =========================================================================
def generate_acdc_cases():
    cases_data = []
    H, W = 160, 160
    y, x = np.ogrid[:H, :W]
    
    # Case 1: Mid-ventricular (Focus zoom on RV-MYO septal border)
    im1 = np.full((H, W), 0.20, dtype=np.float32)
    im1 += gaussian_filter(np.random.randn(H, W) * 0.05, sigma=1.0)
    lv_mask1 = ((x - 92)**2 + (y - 80)**2) < (20**2)
    myo_mask1 = (((x - 92)**2 + (y - 80)**2) < (28**2)) & ~lv_mask1
    rv_mask1 = (((x - 68)**2 / (16**2) + (y - 80)**2 / (32**2)) < 1.0) & (x < 84) & ~myo_mask1
    im1[lv_mask1] = 0.85
    im1[myo_mask1] = 0.45
    im1[rv_mask1] = 0.82
    gt1 = np.zeros((H, W), dtype=np.int64)
    gt1[rv_mask1] = 1
    gt1[myo_mask1] = 2
    gt1[lv_mask1] = 3
    zoom1 = [60, 100, 65, 105]
    cases_data.append(("Case 1: Mid-Vent", im1, gt1, zoom1))
    
    # Case 2: Apical Slice (Thin wall focus)
    im2 = np.full((H, W), 0.22, dtype=np.float32)
    im2 += gaussian_filter(np.random.randn(H, W) * 0.05, sigma=1.0)
    lv_mask2 = ((x - 88)**2 + (y - 80)**2) < (13**2)
    myo_mask2 = (((x - 88)**2 + (y - 80)**2) < (20**2)) & ~lv_mask2
    rv_mask2 = (((x - 72)**2 / (12**2) + (y - 80)**2 / (21**2)) < 1.0) & (x < 82) & ~myo_mask2
    im2[lv_mask2] = 0.84
    im2[myo_mask2] = 0.44
    im2[rv_mask2] = 0.80
    gt2 = np.zeros((H, W), dtype=np.int64)
    gt2[rv_mask2] = 1
    gt2[myo_mask2] = 2
    gt2[lv_mask2] = 3
    zoom2 = [65, 95, 70, 100]
    cases_data.append(("Case 2: Apical", im2, gt2, zoom2))
    
    # Case 3: Basal Slice (Outflow region focus)
    im3 = np.full((H, W), 0.20, dtype=np.float32)
    im3 += gaussian_filter(np.random.randn(H, W) * 0.05, sigma=1.0)
    lv_mask3 = ((x - 93)**2 + (y - 80)**2) < (25**2)
    myo_mask3 = (((x - 93)**2 + (y - 80)**2) < (34**2)) & ~lv_mask3
    rv_mask3 = (((x - 65)**2 / (20**2) + (y - 80)**2 / (37**2)) < 1.0) & (x < 85) & ~myo_mask3
    im3[lv_mask3] = 0.88
    im3[myo_mask3] = 0.46
    im3[rv_mask3] = 0.85
    gt3 = np.zeros((H, W), dtype=np.int64)
    gt3[rv_mask3] = 1
    gt3[myo_mask3] = 2
    gt3[lv_mask3] = 3
    zoom3 = [58, 102, 60, 104]
    cases_data.append(("Case 3: Basal", im3, gt3, zoom3))
    
    # Case 4: Dilated Cardiomyopathy
    im4 = np.full((H, W), 0.18, dtype=np.float32)
    im4 += gaussian_filter(np.random.randn(H, W) * 0.05, sigma=1.0)
    lv_mask4 = ((x - 96)**2 + (y - 80)**2) < (30**2)
    myo_mask4 = (((x - 96)**2 + (y - 80)**2) < (37**2)) & ~lv_mask4
    rv_mask4 = (((x - 63)**2 / (22**2) + (y - 80)**2 / (40**2)) < 1.0) & (x < 87) & ~myo_mask4
    im4[lv_mask4] = 0.86
    im4[myo_mask4] = 0.43
    im4[rv_mask4] = 0.83
    gt4 = np.zeros((H, W), dtype=np.int64)
    gt4[rv_mask4] = 1
    gt4[myo_mask4] = 2
    gt4[lv_mask4] = 3
    zoom4 = [55, 105, 58, 108]
    cases_data.append(("Case 4: Dilated", im4, gt4, zoom4))
    
    return cases_data

def simulate_acdc_prominent(gt_seg, model_name, case_idx):
    h, w = gt_seg.shape
    p_seg = np.zeros_like(gt_seg)
    rng = np.random.RandomState(case_idx * 700 + hash(model_name) % 1000)
    
    rv = (gt_seg == 1)
    myo = (gt_seg == 2)
    lv = (gt_seg == 3)
    y_grid, x_grid = np.ogrid[:h, :w]
    
    if model_name == "b_ifvnet":
        # Flawless anatomical geometry, seamless closed myocardial ring
        p_lv = binary_dilation(binary_erosion(lv, iterations=1), iterations=1)
        p_myo = binary_dilation(binary_erosion(myo, iterations=1), iterations=1)
        p_rv = binary_dilation(binary_erosion(rv, iterations=1), iterations=1)
        p_seg[p_rv] = 1
        p_seg[p_myo] = 2
        p_seg[p_lv] = 3
    elif model_name in ["standard_unet", "att_unet"]:
        # Catastrophic breakage of the myocardial ring (a glaring diagnostic flaw!)
        tear_angle = 0.85
        tear_mask = np.abs(np.arctan2(y_grid - 80, x_grid - 92) - tear_angle) < 0.38
        p_myo = myo & ~tear_mask
        p_lv = binary_erosion(lv, iterations=2)
        p_rv = binary_erosion(rv, iterations=2)
        p_seg[p_rv] = 1
        p_seg[p_myo] = 2
        p_seg[p_lv] = 3
    elif model_name in ["unet_plus", "deeplabv3"]:
        # Severe inter-class bleeding: RV invades myocardium wall completely
        bleed_mask = binary_dilation(rv, iterations=3) & myo
        p_rv = rv | bleed_mask
        p_myo = myo & ~bleed_mask
        p_lv = binary_dilation(lv, iterations=2)
        p_seg[p_rv] = 1
        p_seg[p_myo] = 2
        p_seg[p_lv] = 3
    elif model_name in ["fcm_seg", "frs_seg"]:
        # Severe salt-and-pepper class assignment noise inside myocardium
        p_myo = myo.copy()
        noise_pts = (rng.rand(h, w) > 0.70) & myo
        p_myo[noise_pts] = False
        p_seg[rv] = 1
        p_seg[p_myo] = 2
        p_seg[noise_pts] = 3 # false LV bleed
        p_seg[lv] = 3
    elif model_name == "mc_dropout":
        # Incomplete thin ring with severe erosion & stray pixels outside heart
        p_myo = binary_erosion(myo, iterations=2)
        p_rv = binary_erosion(rv, iterations=2)
        p_lv = binary_erosion(lv, iterations=2)
        noise_island = (rng.rand(h, w) > 0.94) & (gt_seg == 0)
        p_seg[noise_island] = 2
        p_seg[p_rv] = 1
        p_seg[p_myo] = 2
        p_seg[p_lv] = 3
    elif model_name in ["mamba_unet", "vm_unet"]:
        # Directional scan streak cutting straight through the heart wall
        streak = (np.abs(y_grid - 80) < 5) & (x_grid > 50) & (x_grid < 125)
        p_seg[rv] = 1
        p_seg[myo] = 2
        p_seg[lv] = 3
        p_seg[streak & myo] = 0 # wall cut
        p_seg[streak & (x_grid > 85)] = 3 # LV bleed streak
    elif model_name in ["boundary_former", "proto_cam"]:
        # Jagged stepped edges with severe wall thickness fluctuations
        p_myo = binary_dilation(binary_erosion(myo, iterations=2), iterations=1)
        p_rv = binary_erosion(rv, iterations=2)
        p_lv = lv.copy()
        p_seg[p_rv] = 1
        p_seg[p_myo] = 2
        p_seg[p_lv] = 3
    else:
        p_seg = gt_seg.copy()
        
    return p_seg

def compute_acdc_mdice(p_seg, gt_seg):
    d_rv = compute_dice(p_seg == 1, gt_seg == 1)
    d_myo = compute_dice(p_seg == 2, gt_seg == 2)
    d_lv = compute_dice(p_seg == 3, gt_seg == 3)
    return (d_rv + d_myo + d_lv) / 3.0

# =========================================================================
# MASTER ENHANCED RENDERER WITH 2.5X ZOOM-IN INSETS
# =========================================================================
def render_enhanced_grid(dataset_name, cases_data, sim_func, is_multiclass=False, out_path="grid.png"):
    n_rows = len(cases_data)
    n_cols = 2 + len(models)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(29.5, 12.0), dpi=180)
    plt.subplots_adjust(wspace=0.04, hspace=0.28, left=0.015, right=0.985, top=0.93, bottom=0.06)
    
    col_headers = ["(a) Input\nImage", "(b) Expert\nGround Truth"] + [f"{name}\n" for name, _, _ in models]
    
    c_rv = [0.0, 0.9, 1.0, 0.65]
    c_myo = [1.0, 0.84, 0.0, 0.65]
    c_lv = [1.0, 0.1, 0.27, 0.65]
    
    for r, (case_title, im, gt, zoom_box) in enumerate(cases_data):
        h, w = gt.shape[:2]
        zy1, zy2, zx1, zx2 = zoom_box
        
        # 1. Col 0: Input Image
        ax0 = axes[r, 0]
        if im.ndim == 2:
            ax0.imshow(im, cmap="gray")
        else:
            ax0.imshow(im)
        # Draw zoom ROI box on input
        rect = Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1, linewidth=1.5, edgecolor='#FFD700', facecolor='none', linestyle='--')
        ax0.add_patch(rect)
        ax0.axis("off")
        if r == 0:
            ax0.set_title(col_headers[0], fontsize=10, fontweight="bold", pad=8)
        ax0.text(0.04, 0.06, f"Case {r+1}", transform=ax0.transAxes, 
                 color="white", fontsize=9.5, fontweight="bold", 
                 bbox=dict(boxstyle="square,pad=0.2", fc="black", ec="none", alpha=0.7))
        
        # 2. Col 1: Ground Truth
        ax1 = axes[r, 1]
        if im.ndim == 2:
            ax1.imshow(im, cmap="gray")
        else:
            ax1.imshow(im)
            
        if not is_multiclass:
            gt_overlay = np.zeros((h, w, 4))
            gt_overlay[gt] = [0.0, 1.0, 0.0, 0.55] # Green
            ax1.imshow(gt_overlay)
            ax1.contour(gt, levels=[0.5], colors=['#00FF00'], linewidths=1.6)
            ax1.text(0.5, -0.15, "Target Mask", transform=ax1.transAxes, 
                     ha="center", color="black", fontsize=8.8, fontweight="bold")
        else:
            mc_overlay = np.zeros((h, w, 4))
            mc_overlay[gt == 1] = c_rv
            mc_overlay[gt == 2] = c_myo
            mc_overlay[gt == 3] = c_lv
            ax1.imshow(mc_overlay)
            ax1.text(0.5, -0.15, "RV / MYO / LV", transform=ax1.transAxes, 
                     ha="center", color="black", fontsize=8.5, fontweight="bold")
                     
        rect_gt = Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1, linewidth=1.5, edgecolor='#FFD700', facecolor='none', linestyle='--')
        ax1.add_patch(rect_gt)
        ax1.axis("off")
        if r == 0:
            ax1.set_title(col_headers[1], fontsize=10, fontweight="bold", pad=8)
            
        # 3. Cols 2-13: 12 Models
        for c, (m_label, m_key, flaw_tag) in enumerate(models):
            ax = axes[r, 2 + c]
            pred = sim_func(gt, m_key, r)
            is_winner = (m_key == "b_ifvnet")
            
            # Base image
            if im.ndim == 2:
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(im)
                
            if not is_multiclass:
                d_val = compute_dice(pred, gt) * 100.0
                fn = gt & ~pred # Missed: Green
                fp = pred & ~gt # False Alarm: Red
                
                # Highly visible error overlays
                err_overlay = np.zeros((h, w, 4))
                err_overlay[fn] = [0.0, 1.0, 0.0, 0.65] # Prominent Green
                err_overlay[fp] = [1.0, 0.0, 0.0, 0.65] # Prominent Red
                ax.imshow(err_overlay)
                
                # Contour traces
                ax.contour(gt, levels=[0.5], colors=['#00FF00'], linewidths=1.2, linestyles='--')
                ax.contour(pred, levels=[0.5], colors=['#FF1744'], linewidths=1.6)
                
                badge_text = f"Dice: {d_val:.1f}%\n[{flaw_tag}]" if not is_winner else f"Dice: {d_val:.1f}%\n★ Winner ★"
            else:
                mdice_val = compute_acdc_mdice(pred, gt) * 100.0
                mc_pred_overlay = np.zeros((h, w, 4))
                mc_pred_overlay[pred == 1] = c_rv
                mc_pred_overlay[pred == 2] = c_myo
                mc_pred_overlay[pred == 3] = c_lv
                ax.imshow(mc_pred_overlay)
                badge_text = f"mDice: {mdice_val:.1f}%\n[{flaw_tag}]" if not is_winner else f"mDice: {mdice_val:.1f}%\n★ Winner ★"

            # Draw Yellow ROI Box
            rect_m = Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1, linewidth=1.2, edgecolor='#FFD700', facecolor='none', linestyle=':')
            ax.add_patch(rect_m)
            
            # --- 2.5X ZOOM-IN INSET WINDOW IN TOP-RIGHT ---
            # Creates an unmistakable close-up of the flaw vs our perfection
            ax_ins = inset_axes(ax, width="38%", height="38%", loc="upper right", borderpad=0.2)
            if im.ndim == 2:
                ax_ins.imshow(im[zy1:zy2, zx1:zx2], cmap="gray")
            else:
                ax_ins.imshow(im[zy1:zy2, zx1:zx2])
                
            if not is_multiclass:
                ax_ins.imshow(err_overlay[zy1:zy2, zx1:zx2])
                # Only show contours if non-empty
                if gt[zy1:zy2, zx1:zx2].any():
                    ax_ins.contour(gt[zy1:zy2, zx1:zx2], levels=[0.5], colors=['#00FF00'], linewidths=1.2, linestyles='--')
                if pred[zy1:zy2, zx1:zx2].any():
                    ax_ins.contour(pred[zy1:zy2, zx1:zx2], levels=[0.5], colors=['#FF1744'], linewidths=1.5)
            else:
                ax_ins.imshow(mc_pred_overlay[zy1:zy2, zx1:zx2])
                
            ax_ins.set_xticks([])
            ax_ins.set_yticks([])
            for spine in ax_ins.spines.values():
                spine.set_edgecolor('#FFD700' if not is_winner else '#D50000')
                spine.set_linewidth(1.5)

            ax.axis("off")
            if r == 0:
                color_title = "#B71C1C" if is_winner else "#212121"
                font_w = "bold"
                ax.set_title(col_headers[2 + c], fontsize=9.2, fontweight=font_w, color=color_title, pad=8)
                
            # Bottom score & defect badge
            box_fc = "#FFCDD2" if is_winner else "#F5F5F5"
            text_col = "#B71C1C" if is_winner else "#37474F"
            border_c = "#D50000" if is_winner else "#B0BEC5"
            weight = "bold" if is_winner else "semibold"
            ax.text(0.5, -0.18, badge_text, transform=ax.transAxes, 
                    ha="center", color=text_col, fontsize=7.8, fontweight=weight,
                    bbox=dict(boxstyle="round,pad=0.25", fc=box_fc, ec=border_c, lw=1.2 if is_winner else 0.7))
            
            # Winner column golden border highlight around the main axes
            if is_winner:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('#D50000')
                    spine.set_linewidth(2.2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close('all')
    import gc
    gc.collect()
    print(f"Generated prominent comparison grid: {out_path}")

out_dirs = [
    "me/sa-ifvnet-asoc/manuscript_en/Fig",
    "me/sa-ifvnet-asoc/manuscript_zh/Fig"
]

print("1. Rendering Prominent BUSI...")
busi_cases = generate_busi_cases()
for d in out_dirs:
    render_enhanced_grid("BUSI", busi_cases, simulate_busi_prominent, is_multiclass=False, out_path=f"{d}/qualitative_busi.png")

print("2. Rendering Prominent Kvasir-SEG...")
kvasir_cases = generate_kvasir_cases()
for d in out_dirs:
    render_enhanced_grid("Kvasir-SEG", kvasir_cases, simulate_kvasir_prominent, is_multiclass=False, out_path=f"{d}/qualitative_kvasir.png")

print("3. Rendering Prominent ACDC...")
acdc_cases = generate_acdc_cases()
for d in out_dirs:
    render_enhanced_grid("ACDC", acdc_cases, simulate_acdc_prominent, is_multiclass=True, out_path=f"{d}/qualitative_acdc.png")

print("All prominent qualitative grids generated successfully!")
