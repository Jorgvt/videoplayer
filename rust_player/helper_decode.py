import sys
import av
import cv2

if len(sys.argv) < 2:
    sys.exit(1)

path = sys.argv[1]
stdout_bin = sys.stdout.buffer

try:
    container = av.open(path, options={'hwaccel': 'cuda'})
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        img = frame.to_ndarray(format='bgr24')
        stdout_bin.write(img.tobytes())
except Exception:
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        stdout_bin.write(frame.tobytes())
    cap.release()
