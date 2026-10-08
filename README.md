# Cyclone Intensity Prediction AI (Cyclone Core)

An automated, multi-modal machine learning platform designed to identify, analyze, and predict the intensity of tropical cyclones directly from raw multi-spectral satellite imagery. Engineered for the Smart India Hackathon (SIH), the system bridges deep learning computer vision with generative AI to output both numerical intensity forecasts and automated, standardized public warning advisories without manual intervention.

---

## 📌 Problem Alignment & Objectives

Traditional cyclone intensity estimation relies heavily on subjective manual image cropping and cloud pattern matching. During rapid intensification events, human latency can delay life-saving emergency alerts.
* **Standardized Meteorology:** Completely eliminates the reliance on manual image cropping and subjective human interpretation of storm signatures.
* **Accelerated Response:** Delivers instant, deterministic storm classifications and intensity calculations crucial for rapid disaster response deployment.

---

## 🌟 Innovation & Uniqueness of the Solution

* **Multi-Modal AI Architecture (Vision + LLM):** Bridges advanced Computer Vision (YOLOv8 + MobileNetV2 + ResNet-50) with Generative AI (LLMs), creating a complete sequence from raw satellite telemetry to natural language public advisories.
* **Pre-Regression Spatial Gating:** Forces data through a strict multi-model gatekeeper system to isolate and crop the eyewall before executing heavy regression math, guaranteeing hyper-localized, noise-free readings.
* **Explainable AI (XAI) for Mission-Critical Trust:** Live PyTorch attention hooks render thermal heatmaps over convective cloud structures, allowing meteorologists to visually verify exactly which atmospheric features drove the wind speed calculation.
* **Zero-Middleware SACHET Integration:** Natively serializes LLM reports into OASIS CAP v1.2 XML payloads for automated, instant transmission directly to the NDMA SACHET emergency alert gateway.

---

## 🧠 AI & Deep Learning Architecture

* **Ultralytics (YOLOv8) - The Primary Gatekeeper & Cropper:** The lightweight YOLOv8-nano (`yolov8n.pt`) performs two mandatory functions:
  1. **Initial Object Detection:** It screens the uploaded image for standard COCO objects (like humans, vehicles, or regular landscapes) to immediately reject obvious non-satellite images.
  2. **Eye Cropping:** Once confirmed as a valid image, it calculates the spatial bounding box coordinates to dynamically crop the cyclone's exact eyewall.
* **Torchvision (MobileNetV2) - The Texture Scanner:** Before the cropped image can reach the main ResNet-50 model, MobileNetV2 acts as a secondary security gatekeeper. It scans the pixel distribution and cloud texture to verify the image is authentic meteorological satellite data and filters out adversarial noise.
* **Torchvision (ResNet-50) - The Main Regression Engine:** Only validated, properly cropped cyclone images that pass *both* the YOLOv8 and MobileNetV2 filters are allowed to pass into this custom network. It computes the continuous, precise maximum sustained wind speeds (knots) and central atmospheric pressure (hPa).
* **Generative Advisory Engine (LLM):** Ingests the calculated AI metrics to synthesize situational, multi-tier disaster response advisories.

---

## 📂 Repository File Structure

```text
CYCLON PREDICTION/
├── backend/
│   ├── llm_intel.py               # Generates natural language advisories & OASIS CAP v1.2 XML
│   ├── main.py                    # FastAPI application entry point and inference endpoints
│   ├── requirements.txt           # Python backend dependencies
│   └── run.py                     # Subprocess server launcher and process manager
├── frontend/
│   ├── css/
│   │   └── style.css              # Custom UI stylesheet with modern glassmorphism styling
│   ├── js/
│   │   └── script.js              # Client-side logic, telemetry parsing, and UI updates
│   ├── earth-galaxy-elements.jpg  # UI graphical asset / background theme
│   └── index.html                 # Main operator dashboard interface
├── injection/
│   ├── image1.jpg                 # Sample satellite telemetry test frame
│   ├── inject.py                  # Automated pipeline stress-testing script
│   ├── odisha_storm_gps.jpg       # Historical cyclone satellite sample with GPS telemetry
│   └── storm.jpg                  # Secondary test storm sample
├── notebooks/
│   ├── cyclone-project.ipynb      # Kaggle/Jupyter training notebook (ResNet training & metrics)
│   └── cyclone-project.py         # Python script export of the model training pipeline
├── yolov8n.pt                     # Pretrained YOLOv8-nano weights for spatial bounding
├── .gitignore                     # Excludes virtual environments, API keys, caches, and heavy weights
└── README.md                      # Comprehensive project documentation
