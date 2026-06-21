"""
NEOSHOP — Product & Theft Detection System
==========================================
Main detection script. Runs on your laptop during development,
and on Raspberry Pi 4 in production.

LOGIC:
  Camera watches the cart area continuously.
  When a hand is detected holding/moving a product:
    → Check if a barcode scan happened recently (within SCAN_TIMEOUT seconds)
    → If YES → OK, product was scanned
    → If NO  → Issue WARNING to customer
    → If WARNING ignored for THEFT_CONFIRM_SECONDS → THEFT CONFIRMED
       → Activate brake (servo-lock via GPIO on Pi)
       → Send alert to owner dashboard via WebSocket

HOW TO RUN:
  python 2_theft_detection.py

CONTROLS (while running):
  S — simulate a barcode scan (for testing)
  R — reset scan history
  Q — quit

REQUIREMENTS:
  pip install ultralytics opencv-python supervision
"""

import cv2
import time
import threading
import json
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import numpy as np

# ============================================================
# CONFIGURATION — adjust these to tune the system
# ============================================================

MODEL_PATH = "models/best.pt"       # Path to trained model
CAMERA_INDEX = 0                     # 0 = default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.45          # Min confidence to count a detection
IOU_THRESHOLD = 0.45                 # NMS IoU threshold

# Timing
SCAN_TIMEOUT = 8.0        # Seconds: how long after a scan is a product "OK"
WARNING_HOLD = 3.0        # Seconds: hand must hold product before warning fires
THEFT_CONFIRM = 6.0       # Seconds: if warning ignored this long → theft confirmed
COOLDOWN = 10.0           # Seconds: minimum time between two alerts

# Classes — must match your trained model's class names
# For COCO pretrained (no custom training), we map COCO IDs to our labels
PRODUCT_CLASSES = {
    'bottle', 'cup', 'bowl', 'banana', 'apple',
    'orange', 'carrot', 'hot dog', 'pizza', 'cake',
    'book', 'scissors', 'toothbrush', 'cell phone',
    'remote', 'vase', 'clock',
    # Custom trained classes (if you trained with Roboflow):
    'can', 'box', 'bag', 'product'
}
HAND_CLASSES = {'hand'}

# Colors (BGR)
COLOR_OK      = (34, 197, 94)    # Green
COLOR_WARNING = (0, 140, 255)    # Orange
COLOR_THEFT   = (0, 0, 220)      # Red
COLOR_SCAN    = (255, 215, 0)    # Gold
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0, 0, 0)


# ============================================================
# SCAN TRACKER — simulates integration with barcode scanner
# ============================================================

class ScanTracker:
    """
    Tracks barcode scan events.
    In production this connects to the barcode scanner via the FastAPI backend.
    In testing, press 'S' to simulate a scan.
    """
    def __init__(self):
        self.last_scan_time = None
        self.scan_count = 0
        self._lock = threading.Lock()

    def register_scan(self, barcode: str = "MANUAL"):
        with self._lock:
            self.last_scan_time = time.time()
            self.scan_count += 1
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"[SCAN] ✓ Barcode scanned: {barcode} at {ts}")

    def was_scanned_recently(self) -> bool:
        with self._lock:
            if self.last_scan_time is None:
                return False
            return (time.time() - self.last_scan_time) < SCAN_TIMEOUT

    def seconds_since_scan(self) -> float:
        with self._lock:
            if self.last_scan_time is None:
                return float('inf')
            return time.time() - self.last_scan_time

    def reset(self):
        with self._lock:
            self.last_scan_time = None
            print("[SCAN] Reset — scan history cleared")


# ============================================================
# INTERACTION DETECTOR — core logic for hand + product
# ============================================================

class InteractionDetector:
    """
    Determines if a hand is interacting with a product.
    Uses bounding box overlap (IoU) between hand and product detections.
    """

    @staticmethod
    def compute_iou(box1, box2) -> float:
        """Compute Intersection over Union between two boxes [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def boxes_overlap(box1, box2, min_iou: float = 0.05) -> bool:
        """
        Return True if boxes overlap more than min_iou.
        We use a low threshold (0.05) since a hand partially overlapping
        a product is enough to count as interaction.
        """
        return InteractionDetector.compute_iou(box1, box2) >= min_iou

    @staticmethod
    def find_interactions(hand_boxes, product_boxes):
        """
        Find all (hand_idx, product_idx) pairs that overlap.
        Returns list of tuples.
        """
        interactions = []
        for hi, hbox in enumerate(hand_boxes):
            for pi, pbox in enumerate(product_boxes):
                if InteractionDetector.boxes_overlap(hbox, pbox):
                    interactions.append((hi, pi))
        return interactions


# ============================================================
# THEFT DETECTOR — state machine
# ============================================================

class TheftDetector:
    """
    State machine:
      IDLE → HAND_WITH_PRODUCT → WARNING → THEFT_CONFIRMED
    """

    STATE_IDLE      = "idle"
    STATE_HOLDING   = "holding"    # Hand detected with product, within scan window
    STATE_WARNING   = "warning"    # No scan, warning shown on screen
    STATE_THEFT     = "theft"      # Theft confirmed — activate brake

    def __init__(self, scan_tracker: ScanTracker):
        self.scan_tracker = scan_tracker
        self.state = self.STATE_IDLE

        self.holding_start: float = None    # When hand first detected with product
        self.warning_start: float = None    # When warning first issued
        self.last_alert_time: float = 0.0   # Cooldown tracking

        self._alert_callbacks = []
        self._lock = threading.Lock()

        self.fps_history = deque(maxlen=30)
        self.frame_count = 0

    def add_alert_callback(self, fn):
        """Register a function to call when theft is confirmed."""
        self._alert_callbacks.append(fn)

    def _fire_alert(self, event_type: str, data: dict):
        for cb in self._alert_callbacks:
            threading.Thread(target=cb, args=(event_type, data), daemon=True).start()

    def process_frame(self, frame: np.ndarray, detections) -> tuple:
        """
        Main processing function called for every frame.

        Args:
            frame: BGR frame from camera
            detections: ultralytics Results object

        Returns:
            (annotated_frame, state_label, info_dict)
        """
        t_start = time.time()
        self.frame_count += 1

        # --- Parse detections ---
        hand_boxes    = []
        product_boxes = []
        product_labels = []

        if detections and len(detections) > 0:
            result = detections[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id   = int(box.cls)
                    cls_name = result.names[cls_id].lower()
                    conf     = float(box.conf)
                    xyxy     = box.xyxy[0].cpu().numpy()

                    if cls_name in HAND_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                        hand_boxes.append(xyxy)
                    elif cls_name in PRODUCT_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                        product_boxes.append(xyxy)
                        product_labels.append((cls_name, conf))

        # --- Find hand-product interactions ---
        interactions = InteractionDetector.find_interactions(hand_boxes, product_boxes)
        hand_holding_product = len(interactions) > 0

        # --- State machine ---
        now = time.time()
        recently_scanned = self.scan_tracker.was_scanned_recently()

        with self._lock:
            if not hand_holding_product:
                # No interaction — reset to idle (but keep warning state briefly)
                if self.state not in (self.STATE_THEFT,):
                    self.state = self.STATE_IDLE
                    self.holding_start = None

            elif recently_scanned:
                # Hand with product, but scan happened recently → OK
                self.state = self.STATE_HOLDING
                self.holding_start = self.holding_start or now
                self.warning_start = None

            else:
                # Hand with product, NO recent scan
                if self.holding_start is None:
                    self.holding_start = now

                held_for = now - self.holding_start

                if held_for < WARNING_HOLD:
                    # Still within grace period
                    self.state = self.STATE_HOLDING

                elif self.warning_start is None:
                    # Grace period over → issue warning
                    self.state = self.STATE_WARNING
                    self.warning_start = now
                    print(f"[WARN] ⚠️  Unscanned product detected! Warning issued at {datetime.now().strftime('%H:%M:%S')}")
                    self._fire_alert("warning", {
                        "timestamp": datetime.now().isoformat(),
                        "products": [p[0] for p in product_labels],
                    })

                else:
                    warned_for = now - self.warning_start
                    if warned_for >= THEFT_CONFIRM:
                        # Warning ignored → theft confirmed
                        if (now - self.last_alert_time) >= COOLDOWN:
                            self.state = self.STATE_THEFT
                            self.last_alert_time = now
                            print(f"[THEFT] 🚨 THEFT CONFIRMED at {datetime.now().strftime('%H:%M:%S')}")
                            self._fire_alert("theft_confirmed", {
                                "timestamp": datetime.now().isoformat(),
                                "products": [p[0] for p in product_labels],
                                "held_for_seconds": round(now - self.holding_start, 1),
                            })
                    else:
                        self.state = self.STATE_WARNING

            current_state = self.state

        # --- Draw annotations ---
        annotated = self._draw(
            frame, hand_boxes, product_boxes, product_labels,
            interactions, current_state, recently_scanned
        )

        # --- FPS ---
        elapsed = time.time() - t_start
        self.fps_history.append(1.0 / (elapsed + 1e-6))
        fps = sum(self.fps_history) / len(self.fps_history)

        cv2.putText(annotated, f"FPS: {fps:.1f}",
                    (FRAME_WIDTH - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_SCAN, 2)

        info = {
            "state": current_state,
            "hands": len(hand_boxes),
            "products": len(product_boxes),
            "interactions": len(interactions),
            "recently_scanned": recently_scanned,
            "fps": round(fps, 1),
        }

        return annotated, current_state, info

    def _draw(self, frame, hand_boxes, product_boxes, product_labels,
              interactions, state, recently_scanned):
        """Draw all bounding boxes, labels, and status overlay."""
        canvas = frame.copy()

        # --- Status bar (top) ---
        bar_h = 50
        if state == self.STATE_IDLE:
            bar_color  = (40, 40, 40)
            status_txt = "● Monitoring..."
        elif state == self.STATE_HOLDING and recently_scanned:
            bar_color  = COLOR_OK
            status_txt = "✓ Product Scanned — OK"
        elif state == self.STATE_HOLDING:
            bar_color  = COLOR_WARNING
            status_txt = "⚠  Hand detected — Waiting..."
        elif state == self.STATE_WARNING:
            bar_color  = COLOR_WARNING
            status_txt = "⚠  WARNING: Please scan your product!"
        else:  # THEFT
            bar_color  = COLOR_THEFT
            status_txt = "🚨  THEFT ALERT — Cart locked!"

        cv2.rectangle(canvas, (0, 0), (FRAME_WIDTH, bar_h), bar_color, -1)
        cv2.putText(canvas, status_txt, (12, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_WHITE, 2)

        # --- Draw hand boxes ---
        for box in hand_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(canvas, "HAND", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)

        # --- Draw product boxes ---
        for (box, (label, conf)) in zip(product_boxes, product_labels):
            x1, y1, x2, y2 = map(int, box)
            color = COLOR_OK if recently_scanned else COLOR_WARNING
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            txt = f"{label} {conf:.0%}"
            cv2.putText(canvas, txt, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # --- Highlight interactions (connect hand to product) ---
        for hi, pi in interactions:
            if hi < len(hand_boxes) and pi < len(product_boxes):
                hb = hand_boxes[hi]
                pb = product_boxes[pi]
                hc = (int((hb[0]+hb[2])//2), int((hb[1]+hb[3])//2))
                pc = (int((pb[0]+pb[2])//2), int((pb[1]+pb[3])//2))
                cv2.line(canvas, hc, pc, COLOR_WARNING, 2, cv2.LINE_AA)

        # --- Scan info (bottom bar) ---
        secs = self.scan_tracker.seconds_since_scan()
        if secs == float('inf'):
            scan_txt = "Last scan: —"
        elif secs < SCAN_TIMEOUT:
            scan_txt = f"Last scan: {secs:.1f}s ago  ✓ VALID"
            cv2.putText(canvas, scan_txt,
                        (10, FRAME_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_OK, 2)
        else:
            scan_txt = f"Last scan: {secs:.0f}s ago  ✗ EXPIRED"
            cv2.putText(canvas, scan_txt,
                        (10, FRAME_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WARNING, 2)

        if secs != float('inf') and secs < SCAN_TIMEOUT:
            pass  # already drawn above
        else:
            cv2.putText(canvas, scan_txt,
                        (10, FRAME_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLOR_OK if secs < SCAN_TIMEOUT else (120, 120, 120), 2)

        # --- Controls reminder ---
        cv2.putText(canvas, "[S]=Scan  [R]=Reset  [Q]=Quit",
                    (10, FRAME_HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return canvas


# ============================================================
# ALERT HANDLERS — connect to backend / hardware
# ============================================================

def on_warning(event_type: str, data: dict):
    """Called when unscanned product is detected (warning stage)."""
    if event_type != "warning":
        return
    print(f"\n[ALERT] ⚠️  Warning event:")
    print(f"  Products: {data.get('products', [])}")
    print(f"  Time: {data.get('timestamp')}")
    # TODO: Send WebSocket message to customer display
    # ws.send(json.dumps({"type": "warning", **data}))


def on_theft_confirmed(event_type: str, data: dict):
    """Called when theft is confirmed — activate brake and notify staff."""
    if event_type != "theft_confirmed":
        return
    print(f"\n[ALERT] 🚨 THEFT CONFIRMED:")
    print(f"  Products: {data.get('products', [])}")
    print(f"  Held for: {data.get('held_for_seconds')}s")
    print(f"  Time: {data.get('timestamp')}")

    # ---- Raspberry Pi GPIO (uncomment on Pi) ----
    # import RPi.GPIO as GPIO
    # BRAKE_PIN = 18
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(BRAKE_PIN, GPIO.OUT)
    # GPIO.output(BRAKE_PIN, GPIO.HIGH)  # Activate servo-lock brake
    # time.sleep(0.5)

    # ---- WebSocket alert to backend (uncomment when backend is ready) ----
    # import websockets, asyncio
    # async def send():
    #     async with websockets.connect("ws://localhost:8000/ws/security") as ws:
    #         await ws.send(json.dumps({"type": "theft_confirmed", **data}))
    # asyncio.run(send())

    # ---- Log to file ----
    with open("theft_log.txt", "a") as f:
        f.write(json.dumps({"event": "theft_confirmed", **data}) + "\n")
    print("[LOG] Event saved to theft_log.txt")


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("=" * 55)
    print("  NEOSHOP — Product & Theft Detection System")
    print("=" * 55)

    # --- Load model ---
    import os
    if os.path.exists(MODEL_PATH):
        print(f"[MODEL] Loading custom model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
    else:
        print(f"[MODEL] Custom model not found at {MODEL_PATH}")
        print("[MODEL] Loading YOLOv8n pretrained on COCO (80 classes)")
        print("[MODEL] TIP: Run Colab notebook to get best.pt, then put it in models/")
        model = YOLO("yolov8n.pt")

    print(f"[MODEL] ✓ Classes: {list(model.names.values())[:10]}...")

    # --- Open camera ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        print("  → Check that your webcam is connected")
        print("  → Try changing CAMERA_INDEX to 1 or 2")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM]   ✓ Camera opened: {actual_w}×{actual_h}")

    # --- Set up detector ---
    scan_tracker = ScanTracker()
    detector     = TheftDetector(scan_tracker)
    detector.add_alert_callback(on_warning)
    detector.add_alert_callback(on_theft_confirmed)

    print("\n[READY] System running. Controls:")
    print("  S — simulate barcode scan")
    print("  R — reset scan history")
    print("  Q — quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        # Run YOLOv8 inference
        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            stream=False,
        )

        # Process and draw
        annotated, state, info = detector.process_frame(frame, results)

        # Show window
        cv2.imshow("NEOSHOP — Theft Detection  [Q]=Quit", annotated)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("[EXIT] Shutting down...")
            break
        elif key == ord('s') or key == ord('S'):
            scan_tracker.register_scan("TEST_BARCODE_001")
        elif key == ord('r') or key == ord('R'):
            scan_tracker.reset()
            detector.state = TheftDetector.STATE_IDLE
            detector.holding_start = None
            detector.warning_start = None

    cap.release()
    cv2.destroyAllWindows()
    print("[EXIT] System stopped.")


if __name__ == "__main__":
    main()
