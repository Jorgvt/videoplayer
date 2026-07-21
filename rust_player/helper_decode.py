import sys
import cv2

if len(sys.argv) < 2:
    sys.exit(1)

path = sys.argv[1]
cap = cv2.VideoCapture(path)

stdout_bin = sys.stdout.buffer

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    stdout_bin.write(frame.tobytes())

cap.release()
