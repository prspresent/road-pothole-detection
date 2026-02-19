# CPU-Only Real-Time Road Anomaly Detection on Embedded ARM Devices

## Overview
This project presents a real-time road anomaly detection system deployed on Raspberry Pi 5 using CPU-only inference.

The system detects:
- Potholes
- Speed-breakers
- Unpaved roads

The model is optimized for ARM Cortex-A76 architecture using ONNX Runtime without GPU, TPU, or NPU acceleration.

---

## Hardware
- Raspberry Pi 5 (ARM Cortex-A76)
- 8 GB RAM
- USB Camera

---

## Software Stack
- Python 3.11
- ONNX Runtime (CPUExecutionProvider)
- OpenCV
- NumPy
- YOLOv5n (exported to ONNX)

---

## Performance
- Pure Inference: ~48 FPS
- End-to-End Pipeline: 22–28 FPS
- RAM Usage: 127 MB
- CPU Utilization: 47%

---

## System Pipeline
1. Frame Capture
2. Resize (320×320)
3. Normalization
4. ONNX Inference
5. Non-Maximum Suppression
6. Bounding Box Rendering

---

## Real-Time Constraint
The system satisfies:
FPS ≥ 15 (real-time threshold)

---

## Results
The system successfully detects:
- Central road potholes
- Edge potholes
- Large clustered anomalies
- Small surface defects

---

## Author
Praveen Saxena  
Indian Institute of Technology Goa
