# ============================================================
# PROJECT 2: Real-Time Object Detection with YOLOv8
# Stack: Ultralytics YOLOv8, OpenCV, FastAPI
# Install: pip install ultralytics fastapi uvicorn python-multipart
# ============================================================

# ---------- Part A: Live Webcam Inference ----------
from ultralytics import YOLO
import cv2
from collections import Counter

def live_detection():
    """Run YOLOv8 on webcam feed with object counting."""
    model = YOLO("yolov8n.pt")  # auto-downloads on first run
    cap   = cv2.VideoCapture(0)

    print("Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, conf=0.4)[0]
        annotated  = results.plot()
        labels     = [model.names[int(c)] for c in results.boxes.cls]
        counts     = Counter(labels)
        count_text = " | ".join(f"{k}: {v}" for k, v in counts.items())
        cv2.putText(annotated, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Object Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- Part B: Fine-tune on Custom Dataset ----------
def fine_tune_custom():
    """
    Fine-tune YOLOv8 on your own dataset from Roboflow.

    Steps:
      1. Go to roboflow.com → create a project → annotate images
      2. Export as YOLOv8 format → get dataset.yaml path
      3. Run this function
    """
    model = YOLO("yolov8n.pt")
    results = model.train(
        data   = "dataset.yaml",   # from Roboflow export
        epochs = 50,
        imgsz  = 640,
        batch  = 16,
        name   = "custom_detector",
        device = "cpu"             # change to 0 for GPU
    )
    print("Training complete. Best model at:", results.save_dir)
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")


# ---------- Part C: FastAPI Endpoint ----------
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI(title="YOLOv8 Detection API")
_model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Upload an image → get detected objects + counts."""
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results     = _model(img, conf=0.4)[0]
    detections  = []
    for box in results.boxes:
        detections.append({
            "class"     : _model.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "bbox"      : [round(v, 1) for v in box.xyxy[0].tolist()]
        })

    counts = Counter(d["class"] for d in detections)
    return JSONResponse({
        "total_objects": len(detections),
        "counts"       : dict(counts),
        "detections"   : detections
    })

@app.get("/health")
def health():
    return {"status": "ok", "model": "yolov8n"}


# ---------- Run ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        live_detection()

# To run API:  python detect.py api
# To run live: python detect.py
