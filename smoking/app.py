from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import numpy as np
import cv2
import io
from PIL import Image

app = Flask(__name__)

# Load YOLO model
model = YOLO("models/best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Nhận ảnh
        if "file" in request.files:
            image = Image.open(request.files["file"].stream).convert("RGB")
        else:
            data = request.get_json()
            if not data or "image" not in data:
                return jsonify({"error": "No image provided"}), 400
            img_bytes = base64.b64decode(data["image"])
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Chuyển ảnh sang numpy
        frame = np.array(image)
        results = model(frame)

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": box.xyxy[0].tolist()
                })

        return jsonify({"detections": detections})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Smoking detector API is running!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001)
