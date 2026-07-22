# ⚙️ VisionForge AI - Industrial Inspection OS (v9.0)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-green?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.25+-red?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Enterprise--Ready-blue?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Enterprise-grade real-time computer vision platform designed for high-throughput automated assembly lines. Blends hybrid deep inference, calibrated uncertainty metrics, and real-time MLOps telemetry.**

---

## 📌 Executive Overview & Core Problem Solved

Modern industrial assembly lines require continuous, microscopic-level quality control. Standard deep learning classification models often suffer from **overconfidence** and fail under harsh plant-floor conditions (e.g., sensor noise, dynamic lighting fluctuations, missing component variations). 

**VisionForge AI Platform** addresses these challenges by delivering an end-to-end **Vision OS (v9.0)** infrastructure that seamlessly combines:
* **Real-time Image Enhancement**: CLAHE contrast correction in LAB color space and adaptive sharpening filters.
* **Hybrid Decision Fusion**: Fusing object detection (YOLOv11/MobileNetV3) with statistical constraint validation.
* **Calibrated Uncertainty**: Temperature-scaled logits to prevent overconfident false positives.
* **Unsupervised PatchCore Engine**: Vector-embedding distance scoring to discover novel/unknown defect patterns.
* **Active Learning & Drift Control**: Automatic human-in-the-loop routing for low-confidence or high-drift predictions.

---

## 📸 Platform Telemetry Dashboard

Below is a live preview of the operational control console analyzing an active PCB board asset stream:

![VisionForge AI Control Console](https://raw.githubusercontent.com/saqer-alabdallat/visionforge-ai-platform/main/docs/dashboard_pcb.png)

---

## 🏗️ End-to-End System Architecture

```text
[ Raw Frame Asset Stream ]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│               Image Enhancement Pipeline               │
│   (Gaussian Denoising ──► LAB CLAHE ──► Gamma LUT)    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               FastAPI Core Inference Kernel            │
├──────────────────────────┬─────────────────────────────┤
│  Supervised Inspection   │    Unsupervised PatchCore   │
│   (Dimensional / OCR)    │   (Vector Anomaly Distance) │
└──────────────────────────┴─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             Temperature Scaling Calibrator             │
│        (Statistical Confidence Smoothing)              │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│         Root Cause Analysis (RCA) & Telemetry          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             Streamlit Industrial Console               │
│       (Active Learning Routing & SPC Analytics)        │
└────────────────────────────────────────────────────────┘
