import cv2
import numpy as np
import onnxruntime as ort

# ---------------- CONFIG ----------------
MODEL_PATH = "best.onnx"
VIDEO_PATH = "rangoon_demo.mp4"

INPUT_SIZE = 320
FRAME_SKIP = 4
CONF_THRESHOLD = 0.15
POTHOLE_THRESHOLD = 0.08
NMS_THRESHOLD = 0.5
# ----------------------------------------

class_names = ["Speed-breaker", "Pothole", "Unpaved-road"]

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video")
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

    # -------- PREPROCESS --------
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    # -------- INFERENCE --------
    outputs = session.run(None, {input_name: img})
    predictions = outputs[0][0]

    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        obj_conf = pred[4]
        if obj_conf < 0.05:
            continue

        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        confidence = obj_conf * class_scores[class_id]

        if class_id == 1:
            if confidence < POTHOLE_THRESHOLD:
                continue
        else:
            if confidence < CONF_THRESHOLD:
                continue

        x_center, y_center, width_box, height_box = pred[:4]

        x = int((x_center - width_box / 2) * w / INPUT_SIZE)
        y = int((y_center - height_box / 2) * h / INPUT_SIZE)
        bw = int(width_box * w / INPUT_SIZE)
        bh = int(height_box * h / INPUT_SIZE)

        boxes.append([x, y, bw, bh])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    # -------- NMS --------
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.05, NMS_THRESHOLD)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                cls = class_ids[i]

                cv2.rectangle(original, (x, y), (x + bw, y + bh),
                              (0, 255, 0), 2)

                cv2.putText(original,
                            class_names[cls],
                            (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

    cv2.imshow("Road Anomaly Detection", original)

    # Press 's' to save screenshot
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("detection_screenshot.png", original)
        print("Screenshot saved as detection_screenshot.png")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
