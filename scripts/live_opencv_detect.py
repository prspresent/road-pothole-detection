import cv2
import time

print("Starting script...")

cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)  # change to 1 if needed
print("Camera opened:", cap.isOpened())

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not read")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Object",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exited cleanly")
