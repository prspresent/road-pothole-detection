import cv2
import time

# Open USB camera (try 0 first, change to 1 if needed)
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Allow camera to warm up
time.sleep(2)

ret, frame = cap.read()

if ret:
    cv2.imwrite("photo.jpg", frame)
    print("Photo captured and saved as photo.jpg")
else:
    print("Error: Could not capture photo")

cap.release()

