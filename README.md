# B-IFVNet: Bi-directionally Coupled Intuitionistic Fuzzy Variational Network for Robust Lesion Segmentation in Medical Ultrasound

Official PyTorch implementation of **B-IFVNet** for robust boundary delineation under acoustic speckle artifacts, posterior acoustic shadows, and low tissue contrast gradients.

---

## 🌟 Key Highlights

- **Atanassov Intuitionistic Fuzzy Uncertainty Modeling**: Formulates boundary ambiguity into a spatial hesitation field $\pi(x)$ driven jointly by deep predictive entropy and physical acoustic intensity gradients.
- **Bi-directional Feature Calibration Gating**: Dynamically suppresses shallow speckle noise along skip connections prior to decoder feature fusion.
- **Closed-Form Intuitionistic Fuzzy Variational Loss (IFV-Loss)**: Derived via continuous Euler-Lagrange variational extremum conditions, replacing rigid binary cross-entropy with smooth geometric manifold constraints.
- **Comprehensive Clinical Validation**: Statistically significant superiority across 11 competitive baselines on the BUSI breast ultrasound benchmark, with demonstrated transferability to Kvasir-SEG (endoscopic polyps) and ACDC (cardiac cine-MRI).

---

## 📂 Repository Structure

```text
B-IFVNet/
├── models/
│   └── models_zoo.py            # B-IFVNet architecture and all 11 benchmark baselines
├── losses/
│   └── ifv_loss.py              # Closed-form Intuitionistic Fuzzy Variational Loss (IFV-Loss)
├── scripts/
│   ├── train_busi_ultrasound.py # 10-fold cross-validation training on BUSI ultrasound
│   ├── train_kvasir_endoscopy.py# Cross-modality training on Kvasir-SEG polyps
│   ├── train_acdc_cardiac_mri.py# Multi-class topology training on ACDC MRI
│   └── run_ablations.py         # Component ablation experiments
├── evaluation/
│   ├── analyze_results.py       # Metrics computation (Dice, IoU, HD95) and Wilcoxon tests
│   └── generate_qualitative_figures.py # Multi-modal qualitative visualization grids
├── requirements.txt             # Environment dependencies
├── LICENSE                      # MIT License
└── README.md
```

---

## 🛠️ Environment Setup

```bash
conda create -n bifvnet python=3.10 -y
conda activate bifvnet
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Training B-IFVNet on BUSI Breast Ultrasound (10-fold CV)
```bash
python scripts/train_busi_ultrasound.py --data_path /path/to/BUSI --epochs 100 --batch_size 16
```

### 2. Evaluating on Endoscopic Polyp Benchmark (Kvasir-SEG)
```bash
python scripts/train_kvasir_endoscopy.py --data_path /path/to/kvasir-seg --epochs 100
```

### 3. Multi-Class Cardiac MRI Benchmark (ACDC)
```bash
python scripts/train_acdc_cardiac_mri.py --data_path /path/to/ACDC --epochs 100
```

---

## 📊 Benchmark Results (BUSI Breast Ultrasound)

| Model | Dice (DSC) | IoU | HD95 (voxels) |
| :--- | :---: | :---: | :---: |
| Standard U-Net (MICCAI'15) | $0.7462 \pm 0.0152$ | — | $31.84 \pm 3.42$ |
| Attention U-Net (MIDL'18) | $0.7508 \pm 0.0188$ | $0.6732 \pm 0.0184$ | $26.02 \pm 3.88$ |
| FCM-Seg (Fuzzy C-Means) | $0.7565 \pm 0.0148$ | — | $29.75 \pm 2.89$ |
| Mamba-UNet (2025 SOTA) | $0.7048 \pm 0.0227$ | $0.6170 \pm 0.0218$ | $34.66 \pm 2.83$ |
| **B-IFVNet (Ours)** | **0.7759 $\pm$ 0.0117** | **0.6917 $\pm$ 0.0099** | **22.38 $\pm$ 2.21** |

---

## 📜 Citation & License

This project is licensed under the MIT License. If you find B-IFVNet helpful for your research, please cite our manuscript.
