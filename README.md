# 🏭 Industrial Defect Detection System
### KaggleHacX '26 — Data Sprint to the Peak

> An end-to-end AI-powered quality assurance system that classifies metal surface defects in real time using **EfficientNetV2B0** transfer learning, with full **Grad-CAM explainability** and an interactive **Streamlit** dashboard.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Tech Stack](#tech-stack)

---

## Overview

Manual quality assurance on industrial metal surfaces is slow, error-prone, and impossible to scale with modern production volumes. This project delivers an automated defect detection system that:

- **Classifies** metal surface images into 5 defect categories instantly
- **Explains** predictions visually via Grad-CAM heatmaps (pixel-level defect localization)
- **Achieves ~98–99% validation accuracy** through optimized transfer learning
- **Deploys** as a user-friendly Streamlit web application

---

## Dataset

| Property | Details |
|---|---|
| **Competition** | KaggleHacX '26 — Data Sprint to the Peak |
| **Dataset** | Synthetic Industrial Metal Surface Defects |
| **Total Images** | 15,000 |
| **Train / Val Split** | 12,000 / 3,000 |
| **Classes** | 5 (perfectly balanced at 20% per class) |
| **Image Size** | 224 × 224 px (RGB) |

**Classes:** `crack` · `hole` · `normal` · `rust` · `scratch`

---

## Model Architecture

**Backbone:** `EfficientNetV2B0` (pretrained on ImageNet)

EfficientNetV2B0 was chosen over prior-generation models for:
- ⚡ **Fused-MBConv** blocks that accelerate training throughput
- 📈 **Higher accuracy ceiling** with a comparable parameter count
- 🔧 **Native mixed_float16 support** for GPU acceleration

**Custom Classification Head:**
```
GlobalAveragePooling2D
→ BatchNormalization
→ Dense(512, activation='swish')
→ BatchNormalization → Dropout(0.5)
→ Dense(256, activation='swish')
→ BatchNormalization → Dropout(0.4)
→ Dense(5, activation='softmax', dtype='float32')
```

> The output layer is explicitly cast to `float32` for numerical stability under mixed precision.

---

## Training Pipeline

Training uses a **two-phase transfer learning** strategy:

| Phase | Epochs | Learning Rate | Unfrozen Layers | Label Smoothing |
|---|---|---|---|---|
| **Phase 1 – Frozen Base** | 3 | `1e-3` | None (head only) | 0.15 |
| **Phase 2 – Fine-Tuning** | 10 | `1e-5` | Last 50 base layers | 0.15 |

### Data Augmentation (Training Only)

| Augmentation | Range | Rationale |
|---|---|---|
| Rotation | ±45° | Rotated surface captures |
| Zoom | ±35% | Defect size variation |
| Width/Height Shift | ±25% | Off-center placements |
| Brightness | 0.5× – 1.5× | Lighting robustness |
| Horizontal/Vertical Flip | Enabled | Orientation-invariance |
| Channel Shift | ±30 | Color temperature variation |
| Shear | ±25% | Perspective distortion |

### Callbacks
- `EarlyStopping` — monitors `val_accuracy`, restores best weights
- `ReduceLROnPlateau` — reduces LR by 0.3× on plateau (min: `1e-7`)
- `ModelCheckpoint` — saves best model as `model.keras`
- **Mixed Precision** (`mixed_float16`) — ~2× GPU throughput on T4/P100

---

## Results

| Metric | Score |
|---|---|
| **Validation Accuracy** | ~98–99% |
| **Macro F1-Score** | ~0.98 |
| **Weighted F1-Score** | ~0.98 |
| **Inference Latency** | < 100ms (GPU) |
| **Explainability** | Grad-CAM (last Conv2D) |

> High first-epoch accuracy (>95%) is expected and verified as legitimate. The synthetic dataset features high-contrast, visually distinct defect patterns that EfficientNetV2B0's ImageNet features discriminate effectively.

---

## Project Structure

```
kaggle/
├── systematic-industrial-metal-surface-detection          # Main training pipeline (Phase 1 + Phase 2)
├── sanity_check.py                                        # Quick visual sanity check on data samples
├── generate_report.py                                     # Auto-generates methodology_report.docx
├── model.keras                                            # Trained model weights (best checkpoint)
├── confusion_matrix.png                                   # Confusion matrix visualization
├── training_history.png                                   # Accuracy & loss curves across all epochs
├── methodology_report.pdf                                 # Auto-generated technical report
```

---

## Setup & Usage

### Prerequisites

```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas python-docx
```

> **Note:** All training scripts are designed to run on **Kaggle Notebooks** with GPU acceleration enabled. Update `BASE_DIR` in each script if running locally.

### 1. Train the Model

```bash
python trainkaggle.py
```

Runs Phase 1 (frozen base) followed by Phase 2 (fine-tuning). Saves the best model to `model.keras` and generates `training_history.png` and `confusion_matrix.png`.

### 2. Evaluate the Model

```bash
python evaluate_model.py
```

Loads `model.keras`, runs inference on the validation set, and outputs:
- Full classification report (per-class Precision, Recall, F1)
- `confusion_matrix.png` — styled heatmap visualization
- `evaluation_metrics.txt` — saved metrics for records/submission

### 3. Run Diagnostics

```bash
python diagnose.py
```

Performs four checks:
1. **Data Leakage** — MD5 hash comparison between train/val sets
2. **Class Distribution** — per-class image counts with imbalance warnings
3. **Pipeline Sanity** — verifies generator class indices, shapes, and pixel ranges
4. **Prediction Sanity** — checks if the model predicts multiple classes (not collapsed)

### 4. Generate Methodology Report

```bash
pip install python-docx
python generate_report.py
```

Auto-generates `methodology_report.docx` — a fully formatted technical report covering the problem statement, preprocessing, model choice, training strategy, validation approach, and final results.

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `trainkaggle.py` | End-to-end training pipeline with 2-phase transfer learning |
| `evaluate_model.py` | Evaluation metrics + confusion matrix visualization |
| `diagnose.py` | Data leakage detection, class distribution, pipeline & prediction checks |
| `sanity_check.py` | Visual sanity check — renders sample images per class |
| `generate_report.py` | Programmatically builds the `.docx` methodology report |

---

## Tech Stack

| Category | Technology |
|---|---|
| **Deep Learning** | TensorFlow / Keras |
| **Model** | EfficientNetV2B0 (ImageNet pretrained) |
| **Data Processing** | NumPy, Pandas, Keras ImageDataGenerator |
| **Visualization** | Matplotlib, Seaborn |
| **Metrics** | scikit-learn |
| **Explainability** | Grad-CAM (last Conv2D layer) |
| **Dashboard** | Streamlit + Plotly |
| **Report Generation** | python-docx |
| **Training Environment** | Kaggle Notebooks (GPU: T4 / P100) |
| **Precision** | Mixed `float16` + `float32` output head |

---

## 🏆 Hackathon Context

**Competition:** KaggleHacX '26 — Data Sprint to the Peak  
**Challenge:** Synthetic Industrial Metal Surface Defects  
**Time Constraint:** 24–48 hour sprint  
**Outcome:** Optimized, explainable, production-ready QA inspector built from scratch within the hackathon window.

---

*Built for KaggleHacX '26 · April 2026*
