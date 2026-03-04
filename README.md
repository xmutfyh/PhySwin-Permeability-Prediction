# PhySwin-Permeability-Prediction
Official implementation for the paper: "Pore-scale experiments on CO₂ displacement drying kinetics in porous media and physics-informed prediction of dynamic permeability evolution"
# PhySwin – Physics-informed Dynamic Permeability Prediction

Official implementation of the paper:

**Pore-scale experiments on CO₂ displacement drying kinetics in porous media and physics-informed prediction of dynamic permeability evolution**

This repo implements **PhySwin**, a Swin Transformer based model that predicts **dynamic permeability evolution** from **pore-scale images** and **process conditions** (Temperature **T**, Flow rate **V**, Pressure drop **ΔP**).  
The design improves robustness under **Out-of-Distribution (OOD)** conditions by explicitly conditioning the visual features on observable physical variables. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Experimental Setup (from the paper)

### Conditions / Groups
We designed progressive experiments with three factors:  
- Outlet number **C**: 1 / 2 / 3  
- Temperature **T**: 30 / 45 / 60 / 75 ℃  
- CO₂ flow rate **V**: 0.5 / 1.0 / 1.5 / 2.0 mL/min :contentReference[oaicite:4]{index=4}

Representative groups (Table 1):
- **C-series**: C1–C3 (vary outlet number)  
- **T-series**: T1–T4 (vary temperature)  
- **V-series**: V1–V4 (vary flow rate) :contentReference[oaicite:5]{index=5}

### Dataset Summary
- Total: **4185** image frames  
- Conditions: **8** (T,V) combinations  
- Temperature range: **30–75 ℃**, Flow rate range: **0.5–2.0 mL/min**  
- Pressure drop range: **1.8975–17.90125** :contentReference[oaicite:6]{index=6}

### Seen / Unseen Split (OOD)
We evaluate with a two-level split:
- Level-1: split by (T,V) conditions into **Seen (6)** and **Unseen/OOD (2)**.
- Level-2: within Seen, random split train/val/test = **2292 / 487 / 500**. :contentReference[oaicite:7]{index=7}

---

## Environment
All experiments in the paper were run on:
- GPU: NVIDIA GeForce RTX 2060 (6GB)
- Framework: PyTorch 2.5.1, CUDA 12.6
- OS: Windows 10 :contentReference[oaicite:8]{index=8}

---

## Usage

> The main model config used in our commands is:
`models/swin_transformer/tiny_224auto.py`

### 1)  PhySwin Training 
python tools/train_ablation.py models/swin_transformer/tiny_224auto.py --tvp-mode TVP
### 2)  PhySwin Inference
python tools/predict_static_seq_ablation.py models/swin_transformer/tiny_224auto.py <ckpt> --use-tvp --tvp-mode TVP
### 3)  PhySwin Inference Example: 
python tools/predict_static_seq_ablation.py models/swin_transformer/tiny_224auto.py --ckpt logs/SwinTransformer/2026-02-02-10-06-56_tvp-TVP/Val_Epoch084-RMSE0.046.pth --infer-dir datasetone3/infer --right-dir datasetone3/right --use-tvp --tvp-mode TVP 

