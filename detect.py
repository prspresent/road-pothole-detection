import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640,480))
    h, w = frame.shape[:2]

    # Lower half = road
    roi = frame[int(h*0.5):h, 0:w]

    cv2.imshow("Full Frame", frame)
    cv2.imshow("Road ROI", roi)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
