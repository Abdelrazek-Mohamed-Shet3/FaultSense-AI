# FaultSense AI — Multi-Domain Spatio-Temporal Deep Learning Fault Inspection

![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%2F%20CUDA%20%2F%20CPU-4E7A65?style=flat-square)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%2B%20SQLite-6E9582?style=flat-square)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-B8886E?style=flat-square)
![Explainability](https://img.shields.io/badge/XAI-Grad--CAM%20%2B%20Localization-B85D4E?style=flat-square)
![Bilingual](https://img.shields.io/badge/UI-English%20%2F%20%D8%A7%D9%84%D8%B9%D8%B1%D8%A8%D9%8A%D8%A9%20RTL-7E6C8F?style=flat-square)

**FaultSense AI** is a complete, end-to-end Deep Learning platform for **classifying, localizing, and reporting faults from both images and videos** across **four unified domains** — packaged with a pure-white, calm zero-blue React web application, Grad-CAM visual explainability, temporal video attention timelines, custom domain fine-tuning, and executive PDF/JSON reports.

> For deep technical diagrams, tensor flow, loss functions, and API contracts, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Core Capabilities

| Capability | Details |
| :--- | :--- |
| **Unified 4-Domain Engine** | Industrial Manufacturing, Infrastructure & Roads, Safety/PPE Compliance, plus a live Custom Domain Trainer |
| **Image + Video Deep Learning** | Spatial CNN (`MobileNetV3-Small` / `ResNet-18` + Squeeze-and-Excitation) + 2-Layer Temporal Transformer Attention |
| **Explainable AI (Grad-CAM)** | Real gradient-weighted attention overlays + contour-based fault bounding boxes + interactive opacity blend slider |
| **Temporal Video Analytics** | Annotated H.264 MP4 output, frame-by-frame fault probability timeline, and top-3 peak fault keyframes |
| **Custom Domain Studio** | Upload custom labeled classes or launch a 1-click demo fine-tune with live epoch telemetry and model hot-swap |
| **Reports & Audit Trail** | Persistent SQLite inspection log + instant ReportLab PDF & JSON inspection reports |
| **Calm Pure-White UI** | Eye-soothing White + Alabaster + Sage / Sand / Terracotta / Dusty Lavender palette (strictly zero blue) with EN/AR RTL switch |

---

## Fault Taxonomy (10 Built-in Classes + Custom)

| Domain | Class ID | Fault / State | Severity Tier |
| :--- | :--- | :--- | :--- |
| **Industrial & PCB** | `industrial_normal` | Intact Industrial Surface | `PASS` |
| **Industrial & PCB** | `industrial_scratch` | Surface Scratch / Abrasion | `WARNING` |
| **Industrial & PCB** | `industrial_crack` | Structural Crack / Fracture | `CRITICAL` |
| **Industrial & PCB** | `industrial_pcb_defect` | PCB Solder / Pad Fault | `CRITICAL` |
| **Infrastructure & Roads** | `infra_normal` | Intact Pavement / Structure | `PASS` |
| **Infrastructure & Roads** | `infra_pothole` | Road Pothole / Spalling | `CRITICAL` |
| **Infrastructure & Roads** | `infra_corrosion` | Surface Corrosion / Rust | `WARNING` |
| **Safety & Action / PPE** | `safety_compliant` | PPE & Zone Compliant | `PASS` |
| **Safety & Action / PPE** | `safety_no_helmet` | Missing Helmet / PPE Fault | `CRITICAL` |
| **Safety & Action / PPE** | `safety_hazard_zone` | Hazard Zone Intrusion | `CRITICAL` |
| **Custom Studio** | `custom_*` | User-defined fine-tuned fault classes | Dynamic |

---

## Verified Benchmark Results (`artifacts/metrics.json`)

Trained on Apple Silicon **`mps`** (`138.4s` total: 8 spatial epochs + 4 temporal video epochs):

| Metric | Score |
| :--- | :--- |
| **Image Test Accuracy** (160 held-out samples) | **100.0%** |
| **Macro F1-Score / Precision / Recall** | **100.0%** |
| **Video Clip Validation Accuracy** (12-frame clips) | **100.0%** |
| **Backbone** | `MobileNetV3-Small` + `SEBlock` + 2-Layer Temporal Transformer |

---

## Project Structure

```text
FaultSense-AI/
├── ARCHITECTURE.md           # Detailed system diagrams, math, and API contract
├── README.md                 # Project overview and quick start
├── requirements.txt          # Python dependencies
├── run.sh                    # Smart launcher (auto-selects Python env with PyTorch/Uvicorn)
├── ml/
│   ├── taxonomy.py           # 4-domain taxonomy, bilingual labels, severity weights
│   ├── model.py              # FaultSenseSpatialEncoder, SEBlock, TemporalAttentionAggregator
│   ├── gradcam.py            # Grad-CAM hooks, zero-blue LUT, fault region localization
│   ├── dataset.py            # Multi-domain image/video synthesis, sample & folder loaders
│   ├── train.py              # 2-stage spatial + temporal training & custom fine-tuning
│   ├── inference.py          # Thread-safe image/video engine & H.264 HUD video renderer
│   └── reporting.py          # ReportLab executive PDF & JSON exporter
├── backend/
│   ├── database.py           # SQLite inspection history persistence
│   └── main.py               # FastAPI server + static React UI hosting
├── frontend/
│   ├── src/                  # React + TypeScript + Tailwind (White & Zero-Blue UI)
│   └── dist/                 # Pre-built production frontend bundle
├── artifacts/
│   ├── checkpoints/          # Pre-trained faultsense_best.pt checkpoint
│   ├── samples/              # 10 sample images + 4 sample H.264 MP4 videos
│   └── metrics.json          # Real training & confusion matrix evaluation metrics
└── tests/
    └── test_e2e.py           # Automated end-to-end API, ML, video & PDF verification suite
```

---

## Quick Start

### 1. Run the Full-Stack App Immediately
The repository ships with a trained checkpoint (`artifacts/checkpoints/faultsense_best.pt`), benchmark samples, and the compiled frontend bundle:

```bash
./run.sh
```

Then open **http://127.0.0.1:8088** in your browser.

### 2. Re-Train the Multi-Domain Model (Optional)
Generates/loads the `1,100`-image multi-domain dataset (`110` per class across `train`/`val`/`test`) and trains both spatial and temporal stages:

```bash
python3 -m ml.train
```

### 3. Run End-to-End Verification Tests
```bash
python3 tests/test_e2e.py
```

### 4. Frontend Development Mode (Optional)
```bash
cd frontend
npm install
npm run dev
```
