# 🚀 CPU-Only Real-Time Road Anomaly Detection  
### Raspberry Pi 5 | YOLOv5n | ONNX Runtime | ARM Cortex-A76

This project implements a **real-time road anomaly detection system** deployed on **Raspberry Pi 5** using **CPU-only inference**.

Unlike conventional solutions that rely on GPU acceleration, this system achieves real-time performance on embedded ARM hardware using optimized ONNX Runtime execution.

---

## 🎯 Detected Anomalies
- Potholes  
- Speed-breakers  
- Unpaved roads  

---

## 🧠 Model
- **Architecture:** YOLOv5n (Nano variant)  
- **Parameters:** 1.86M  
- **Complexity:** 4.5 GFLOPs  
- **Input Resolution:** 320 × 320  
- **Export Format:** ONNX  

### Training Performance
- mAP@0.5: **0.815**
- F1-score: **0.79**
- Precision: 0.808
- Recall: 0.773

---

## 📊 Performance on Raspberry Pi 5

### Pure Inference
- ~20.5 ms per frame  
- **48.79 FPS**

### Full Pipeline (Capture + Preprocess + Inference + NMS + Render)
- ~46 ms latency  
- **21.67 FPS**

### Resource Usage
- Peak RAM: ~125 MB  
- CPU Utilization: ~311% (multi-core execution)

✔ Real-time constraint satisfied (FPS ≥ 15)

---

## 🏗 System Pipeline
1. Frame Capture (USB Camera)  
2. Resize to 320×320  
3. Normalization  
4. ONNX Inference (CPUExecutionProvider)  
5. Non-Maximum Suppression  
6. Bounding Box Rendering  

---

## 💻 Hardware
- Raspberry Pi 5 (ARM Cortex-A76)  
- 8 GB RAM  
- USB Camera  

---

## 🛠 Software Stack
- Python 3.11  
- ONNX Runtime (CPUExecutionProvider)  
- OpenCV  
- NumPy  
- YOLOv5n  

---

## 📦 Dataset
Trained on a publicly available Kaggle road damage dataset:
- Reformatted into 3 unified classes  
- Converted to YOLO format  
- Resized to 320×320 for embedded efficiency  

Focus: **Accuracy + Embedded Deployment Efficiency**

---

## 🔑 Key Contribution
- Real-time CPU-only inference on ARM  
- No GPU / TPU / NPU required  
- Multi-core optimized execution  
- Lightweight model suitable for edge deployment  

---

## 📷 Example Output
Add a sample detection image inside an `images/` folder:

```markdown
![Detection Example](images/sample_output.png)
