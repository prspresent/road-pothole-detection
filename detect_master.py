import cv2
import numpy as np
import onnxruntime as ort
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "best.onnx"
VIDEO_PATH = "ruralRoad_potHoles.mp4"

INPUT_SIZE = 320
FRAME_SKIP = 2            # detect more frequently
CONF_THRESHOLD = 0.04     # lower global threshold
POTHOLE_THRESHOLD = 0.03  # more sensitive for pothole
NMS_THRESHOLD = 0.45
# ----------------------------------------

class_names = ["Speed-breaker", "Pothole", "Unpaved-road"]

# ONNX Session
so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    original = frame.copy()
    h, w, _ = original.shape

    # ---------------- PREPROCESS ----------------
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0

    # ---------------- INFERENCE ----------------
    outputs = session.run(None, {input_name: img})
    predictions = outputs[0][0]

    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:

        x_center, y_center, width_box, height_box = pred[0:4]
        obj_conf = pred[4]
        class_scores = pred[5:]

        class_id = np.argmax(class_scores)
        confidence = obj_conf * class_scores[class_id]

        # More sensitive for potholes
        if class_id == 1:
            if confidence < POTHOLE_THRESHOLD:
                continue
        else:
            if confidence < CONF_THRESHOLD:
                continue

        # Scale back to original image
        x = int((x_center - width_box / 2) * w / INPUT_SIZE)
        y = int((y_center - height_box / 2) * h / INPUT_SIZE)
        bw = int(width_box * w / INPUT_SIZE)
        bh = int(height_box * h / INPUT_SIZE)

        boxes.append([x, y, bw, bh])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    # ---------------- NMS ----------------
    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        score_threshold=0.01,
        nms_threshold=NMS_THRESHOLD
    )

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = f"{class_names[class_ids[i]]} {confidences[i]:.2f}"

            # Big potholes get thicker box
            thickness = 3 if bw * bh > 20000 else 2

            cv2.rectangle(original, (x, y), (x + bw, y + bh), (0, 255, 0), thickness)
            cv2.putText(original, label,
                        (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    cv2.imshow("Road Anomaly Detection", original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
