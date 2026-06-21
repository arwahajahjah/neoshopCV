================================================================
  NEOSHOP — Computer Vision System
  Product & Theft Detection
================================================================

FILES IN THIS FOLDER:
  1_setup_colab.ipynb   → Train the model on Google Colab (GPU)
  2_theft_detection.py  → Main detection system (run on your laptop/Pi)
  3_test_camera.py      → Test that your webcam works
  models/               → Put best.pt here after training
  theft_log.txt         → Auto-created when theft is detected

================================================================
  STEP-BY-STEP GUIDE
================================================================

── STEP 1: Install dependencies ────────────────────────────────

  pip install ultralytics opencv-python supervision

────────────────────────────────────────────────────────────────

── STEP 2: Test your camera ────────────────────────────────────

  python 3_test_camera.py

  ✓ You should see your webcam feed with FPS shown
  ✗ If error: check webcam is connected, or change CAMERA_INDEX
    in 2_theft_detection.py from 0 to 1

────────────────────────────────────────────────────────────────

── STEP 3: Get the trained model ───────────────────────────────

  OPTION A — Use our Colab notebook (recommended):
  1. Go to https://colab.research.google.com
  2. File → Upload notebook → select 1_setup_colab.ipynb
  3. Runtime → Change runtime type → T4 GPU
  4. Run all cells in   order
  5. The last cell downloads best.pt to your computer
  6. Move best.pt into the models/ folder

  OPTION B — Skip training (works right now, no GPU needed):
  The system automatically falls back to YOLOv8n pretrained
  on COCO-80 classes if models/best.pt is not found.
  COCO already detects: bottle, cup, bowl, banana, apple,
  orange, book, scissors, cell phone, and 70 other objects.
  This is fine for testing and demos.

────────────────────────────────────────────────────────────────

── STEP 4: Run the detection system ────────────────────────────

  python 2_theft_detection.py

  While running:
    S → simulate a barcode scan (for testing without scanner)
    R → reset scan history
    Q → quit

────────────────────────────────────────────────────────────────

── HOW IT WORKS ────────────────────────────────────────────────

  The camera watches the cart area continuously.

  Normal flow (no theft):
    Customer picks up item → hand+product detected → S pressed
    (barcode scanned) → green bar → OK

  Theft flow:
    Customer picks up item → hand+product detected
    No scan within 8 seconds → ORANGE WARNING appears on screen
    Customer ignores warning for 6 more seconds →
    RED ALERT fires → brake activates → staff notified

  Timing (all adjustable at top of 2_theft_detection.py):
    SCAN_TIMEOUT    = 8s   (how long a scan stays "valid")
    WARNING_HOLD    = 3s   (grace period before warning fires)
    THEFT_CONFIRM   = 6s   (time before theft confirmed)
    COOLDOWN        = 10s  (minimum time between alerts)

────────────────────────────────────────────────────────────────

── RASPBERRY PI DEPLOYMENT ─────────────────────────────────────

  1. Copy this entire folder to the Pi
  2. Install: pip3 install ultralytics opencv-python supervision
  3. In 2_theft_detection.py:
     - Uncomment the GPIO brake section (~line 250)
     - Uncomment the WebSocket alert section (~line 258)
     - Set BRAKE_PIN to your servo-lock GPIO pin
  4. Run: python3 2_theft_detection.py

  Performance on Raspberry Pi 4:
    With best.pt (YOLOv8n): ~8-12 FPS (acceptable)
    For higher FPS: export model to ncnn or tflite format

  To export to faster format:
    from ultralytics import YOLO
    model = YOLO('models/best.pt')
    model.export(format='ncnn')   # ~2x faster on Pi

────────────────────────────────────────────────────────────────

── INTEGRATION WITH BACKEND ────────────────────────────────────

  The WebSocket alert in on_theft_confirmed() sends to:
    ws://localhost:8000/ws/security

  Payload example:
    {
      "type": "theft_confirmed",
      "timestamp": "2026-05-09T14:32:11",
      "products": ["bottle", "can"],
      "held_for_seconds": 15.3
    }

  Your FastAPI backend should have a /ws/security endpoint
  that forwards the alert to the owner dashboard and
  security panel.

================================================================
  QUESTIONS? Check the project report Chapter 2.3.3
================================================================
