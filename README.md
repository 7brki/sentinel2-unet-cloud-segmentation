# 🛰️ Autonomous Cloud Segmentation in Sentinel-2 Satellite Imagery using U-Net

## 📌 Project Overview
This project is an end-to-end Deep Learning pipeline designed to detect cloud masses in multispectral (7-Band) images obtained from the European Space Agency's (ESA) Sentinel-2 satellites. The project aims to programmatically repair the hardware-induced flaws of standard satellite sensors and establish an autonomous mapping system.

## 🎯 Problem Statement and Motivation
In traditional remote sensing systems and Sentinel-2's native SCL (Scene Classification Layer) band, cloud detection is generally performed using **pixel-based thresholding** or basic machine learning algorithms like Random Forest. 

However, these methods suffer from "Spatial Blindness":
1. **False Positives:** Limestone terrains, urban concrete structures, and bright sand dunes are frequently misclassified as "clouds" by satellites due to their optical reflectance.
2. **Salt & Pepper Noise:** Because pixel-based systems fail to grasp the organic integrity (geometry) of clouds, the resulting cloud masks are fragmented and noisy.

## 💡 Solution Architecture: Why U-Net?
To overcome these issues, one-dimensional pixel analysis was abandoned, and a **U-Net** deep learning architecture was built from scratch to read pixels alongside their Spatial Context. Thanks to U-Net's "skip connections," the model successfully filtered out the coarse noise of the satellite and achieved much smoother cloud geometries that align with physical reality.

## ⚙️ System Architecture and Data Pipeline
The project is built on a fully autonomous pipeline that eliminates manual data downloading processes.
* **Data Source:** Microsoft Planetary Computer API (Sentinel-2 L2A).
* **Spectral Bands:** 7 different wavelengths (B01, B02, B03, B04, B08, B11, B12) were used to convert the physical properties of clouds (aerosol, water vapor, thermal reflectance) into multi-dimensional tensors.
* **Hardware Acceleration:** The system autonomously detects CUDA (GPU) availability via PyTorch and executes the entire training process on the graphics processing unit.

## 🚧 Overcoming Engineering Bottlenecks
During the construction of this model, two critical bottlenecks frequently encountered in industrial deep learning projects were identified and resolved at the architectural level:

### 1. Data Poisoning and "Dying ReLU" Lockup
* **Problem:** Extremely high raw satellite reflectance values (e.g., 8000+) paralyzed the model weights at the start of training, leading to a stagnant Loss rate (brain death).
* **Solution:** The data was normalized to a 0-1 range, and **`BatchNorm2d`** layers were integrated into every convolutional layer of the U-Net architecture to regulate the data flow.

### 2. Overfitting and the "Batch Size = 1" Trap
* **Problem:** When the model was trained on a single map, it reached 99% accuracy but crashed with a Jaccard (IoU) score of 0.0003 on an unseen geography, proving it had strictly memorized the data.
* **Solution:** A dynamic **"On-the-fly Data Augmentation"** generator was added to the training loop using `Torchvision Transforms`. By randomly rotating and flipping the images in every epoch, the model was forced to learn "cloud geometry" rather than memorizing exact pixel locations.

## 📊 Test Results and Metrics
The armored U-Net model, rescued from rote memorization, was tested on completely new geographies with entirely different topography and lighting angles that it had never seen during training.
* **Final Performance:** The model autonomously reached a **65% - 67% Jaccard Index (IoU)** band in foreign regions, comfortably surpassing the 0.50 industry threshold accepted for professional detection.
* **Visual Output:** Sensor-induced "Salt and Pepper" noise was entirely filtered out, resulting in holistic and consistent cloud maps. *(Sample test outputs can be viewed in the repository images).*

## 🚀 Future Work
The current pipeline has successfully completed the Proof of Concept (PoC) phase by operating on a single geographic coordinate. The next phase is to expand the Dataloader infrastructure to feed the model with a massive dataset consisting of multiple topographies (e.g., deserts, oceans, snowy mountains) and push the IoU score beyond the 85%+ level.

## 💻 Installation and Usage
To test the project in your own environment, the following libraries must be installed:
```bash
pip install pystac-client planetary-computer rasterio torch torchvision matplotlib numpy
