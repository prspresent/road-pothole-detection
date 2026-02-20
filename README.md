# 🇮🇳 Bharat AI-SoC Student Challenge  
## Real-Time Road Anomaly Detection on ARM SoC (CPU-Only)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![SoC](https://img.shields.io/badge/SoC-ARM%20Cortex--A76-green)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Project Summary

This project demonstrates a real-time road anomaly detection system running entirely on an embedded ARM SoC using CPU-only inference.

The system is deployed on a Raspberry Pi 5 and detects:

- Potholes  
- Speed-breakers  
- Unpaved roads  

No GPU, TPU, or NPU acceleration is used.

The goal was to prove that real-time AI inference is possible on low-power embedded hardware.

---

## 🎯 Challenge Focus

Most computer vision systems rely on GPUs for performance.

This project solves the harder problem:

- Real-time performance on CPU  
- Embedded ARM deployment  
- Low memory footprint  
- Multi-core optimization  

---

## 🧠 Model Details

- Model: YOLOv5n (Nano variant)
- Parameters: 1.86M
- Input Size: 320 × 320
- Format: ONNX
- Inference Engine: ONNX Runtime (CPUExecutionProvider)

The nano model was selected to balance performance and efficiency for embedded systems.

---

## 📊 Performance on Raspberry Pi 5

### Pure Inference
- ~20.5 ms per frame  
- ~48 FPS  

### Full Pipeline (Capture → Render)
- ~46 ms latency  
- ~21 FPS sustained  

### Resource Usage
- Peak RAM: ~125 MB  
- CPU Usage: ~311% (multi-core execution)

The system consistently runs above the real-time threshold of 15 FPS.

---

## 🏗 System Pipeline

1. Frame capture  
2. Resize to 320×320  
3. Normalization  
4. ONNX inference  
5. Non-Maximum Suppression  
6. Bounding box rendering  

Optimized for ARM Cortex-A76 multi-core execution.

---

## 📦 Dataset

Trained using a public Kaggle road damage dataset.

Processing steps:
- Merged categories into 3 practical classes  
- Converted to YOLO format  
- Resized to 320×320  
- Tuned confidence threshold for deployment  

---

## 💻 Hardware

- Raspberry Pi 5  
- ARM Cortex-A76 SoC  
- 8GB RAM  
- USB Camera  

---

## 🛠 Software Stack

- Python 3.11  
- ONNX Runtime  
- OpenCV  
- NumPy  
- YOLOv5n  

---

## ▶️ Installation

git clone https://github.com/prspresent/road-pothole-detection.git
cd road-pothole-detection
python3 -m venv road_env
source road_env/bin/activate
pip install -r requirements.txt
## ▶️ Run

Ensure the following files are present:

- `best.onnx`
- `detect_a.py`
- `rangoon_demo.mp4`

Run:


python detect_a.py

Press **Q** to exit.


## 🇮🇳 Relevance to Bharat AI-SoC

This project demonstrates:

- On-device AI inference  
- Efficient ARM SoC utilization  
- Edge AI deployment  
- Real-time embedded system engineering  

It showcases how AI models can run efficiently on embedded processors without relying on cloud or GPU infrastructure.

---

## 👤 Author

Praveen Saxena  
M.Tech – Electrical Engineering  
Indian Institute of Technology Goa  

---

## 📜 License

MIT License
