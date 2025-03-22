from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import sys
from ultralytics import YOLO
import numpy as np
import traceback
import torch
from torchvision import transforms

app = Flask(__name__)

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "best.pt"
)

model_type = None
model = None

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}", file=sys.stderr)
    print(f"Working directory: {os.getcwd()}", file=sys.stderr)
    print(
        f"Directory contents: {os.listdir(os.path.dirname(MODEL_PATH))}",
        file=sys.stderr,
    )
else:
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}", file=sys.stderr)

        if hasattr(model, "task") and model.task is not None:
            model_type = model.task
            print(f"Detected model type: {model_type}", file=sys.stderr)
        else:
            model_type = "detect"
            print(
                f"Could not determine model type, defaulting to: {model_type}",
                file=sys.stderr,
            )

    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    try:
        img = Image.open(io.BytesIO(img_bytes))

        if img.mode != "RGB":
            img = img.convert("RGB")

        try:
            if model_type == "classify":
                input_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    results = model.predict(source=input_tensor)
            else:
                results = model(img, verbose=False)

        except Exception as e:
            print(f"Primary inference error: {str(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

            try:
                print(
                    "Trying alternative approach 1: predict method...", file=sys.stderr
                )
                results = model.predict(source=img, verbose=False)
            except Exception as e1:
                print(f"Alternative 1 failed: {str(e1)}", file=sys.stderr)

                try:
                    print(
                        "Trying alternative approach 2: direct model...",
                        file=sys.stderr,
                    )
                    if hasattr(model, "model"):
                        temp_path = "/tmp/temp_image.jpg"
                        img.save(temp_path)
                        results = model.predict(source=temp_path, verbose=False)

                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    else:
                        raise Exception("No model.model attribute")
                except Exception as e2:
                    print(f"Alternative 2 failed: {str(e2)}", file=sys.stderr)

                    return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

        predictions = []
        try:
            if model_type == "classify":
                for result in results:
                    if hasattr(result, "probs") and result.probs is not None:
                        top_indices = result.probs.top5
                        top_probs = result.probs.top5conf

                        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                            idx_val = int(idx.item() if torch.is_tensor(idx) else idx)
                            prob_val = float(
                                prob.item() if torch.is_tensor(prob) else prob
                            )

                            cls_name = (
                                result.names[idx_val]
                                if hasattr(result, "names")
                                and result.names
                                and idx_val in result.names
                                else f"class_{idx_val}"
                            )
                            prediction = {
                                "class": cls_name,
                                "class_id": idx_val,
                                "confidence": prob_val,
                                "rank": i + 1,
                            }
                            predictions.append(prediction)
            else:
                for result in results:
                    if hasattr(result, "boxes") and result.boxes is not None:
                        for box in result.boxes:
                            if (
                                hasattr(box, "cls")
                                and hasattr(box, "conf")
                                and hasattr(box, "xyxy")
                            ):
                                cls_id = int(
                                    box.cls[0].item()
                                    if hasattr(box.cls[0], "item")
                                    else box.cls[0]
                                )
                                conf = float(
                                    box.conf[0].item()
                                    if hasattr(box.conf[0], "item")
                                    else box.conf[0]
                                )

                                if hasattr(box.xyxy[0], "tolist"):
                                    bbox = [
                                        float(coord) for coord in box.xyxy[0].tolist()
                                    ]
                                else:
                                    bbox = [float(coord) for coord in box.xyxy[0]]

                                cls_name = (
                                    result.names[cls_id]
                                    if result.names and cls_id in result.names
                                    else f"class_{cls_id}"
                                )

                                prediction = {
                                    "class": cls_name,
                                    "class_id": cls_id,
                                    "confidence": conf,
                                    "bbox": bbox,
                                }
                                predictions.append(prediction)
        except Exception as e:
            print(f"Results processing error: {str(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return jsonify({"error": f"Failed to process results: {str(e)}"}), 500

        return jsonify(
            {"success": True, "model_type": model_type, "predictions": predictions}
        )

    except Exception as e:
        print(f"General error: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500


def run_app():
    app.run(debug=True, host="0.0.0.0", port=5000)
