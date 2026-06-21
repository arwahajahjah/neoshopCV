"""
NEOSHOP — Camera Test
======================
Run this FIRST to verify your camera works before running the main system.
  python 3_test_camera.py
"""

import cv2
import time

print("Testing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera 0 not found. Trying camera 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ No camera found. Check your webcam connection.")
    exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✅ Camera found: {w}×{h}")
print("Press Q to quit\n")

fps_list = []
while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    fps = 1.0 / (time.time() - t0 + 0.001)
    fps_list.append(fps)
    avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])

    cv2.putText(frame, f"Camera OK — FPS: {avg_fps:.1f}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
    cv2.putText(frame, "Press Q to quit",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Camera test complete. Average FPS: {avg_fps:.1f}")
