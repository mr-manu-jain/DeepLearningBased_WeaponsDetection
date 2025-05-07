from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import torch
import io
import base64
from PIL import Image
from pathlib import Path
from typing import List, Dict
from ultralytics import YOLO

app = FastAPI()

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (e.g., upload.html)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Model loading block
MODEL_DIR = Path(__file__).parent / "model"

try:
    yolov5_path = MODEL_DIR / "yolov5.pt"
    yolov8_path = MODEL_DIR / "yolov8.pt"

    if not yolov5_path.exists():
        raise FileNotFoundError(f"{yolov5_path} not found.")
    if not yolov8_path.exists():
        raise FileNotFoundError(f"{yolov8_path} not found.")

    MODELS = {
        "yolov5": torch.hub.load('ultralytics/yolov5', 'custom', path=str(yolov5_path)),
        "yolov8": YOLO(str(yolov8_path)),
    }

    MODELS["yolov5"].eval()
except Exception as e:
    print("❌ Error loading models:", e)
    MODELS = {}

# Helper: safely get model names for enum
def get_model_names():
    return list(MODELS.keys()) if MODELS else ["yolov5"]

# Inference helper
def run_inference(model_name: str, img: Image.Image):
    model = MODELS.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' not found.")

    if model_name == "yolov5":
        results = model(img)
        results.render()
        rendered = Image.fromarray(results.ims[0])
        preds = results.pred[0].tolist()
        class_names = model.names

    elif model_name == "yolov8":
        results = model.predict(img, verbose=False)
        rendered = results[0].plot(pil=True)
        preds = results[0].boxes.data.tolist()
        class_names = results[0].names

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return rendered, preds, class_names

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(Path(__file__).parent / "static" / "upload.html")

@app.post("/predict/")
async def predict(
    files: List[UploadFile] = File(...),
    model_name: str = Query("yolov5", enum=get_model_names())
) -> List[Dict]:
    results_data = []

    for file in files:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        try:
            rendered, preds, class_names = run_inference(model_name, img)
        except Exception as e:
            return [{"filename": file.filename, "error": str(e)}]

        output_buffer = io.BytesIO()
        rendered.save(output_buffer, format="JPEG")
        base64_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        detections = []
        for pred in preds:
            x1, y1, x2, y2, conf, cls = pred
            detections.append({
                "class": class_names[int(cls)],
                "confidence": round(float(conf), 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

        results_data.append({
            "filename": file.filename,
            "model_used": model_name,
            "detections": detections,
            "image_base64": base64_image
        })

    return results_data

@app.post("/predict-all/")
async def predict_all(files: List[UploadFile] = File(...)) -> List[Dict]:
    results_data = []

    for file in files:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        per_model_results = {}

        for model_name in MODELS.keys():
            try:
                rendered, preds, class_names = run_inference(model_name, img)

                output_buffer = io.BytesIO()
                rendered.save(output_buffer, format="JPEG")
                base64_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

                detections = []
                for pred in preds:
                    x1, y1, x2, y2, conf, cls = pred
                    detections.append({
                        "class": class_names[int(cls)],
                        "confidence": round(float(conf), 4),
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                    })

                per_model_results[model_name] = {
                    "detections": detections,
                    "image_base64": base64_image
                }

            except Exception as e:
                per_model_results[model_name] = {
                    "error": str(e)
                }

        results_data.append({
            "filename": file.filename,
            "results": per_model_results
        })

    return results_data

@app.get("/classes/")
async def get_classes(model_name: str = Query("yolov5", enum=get_model_names())) -> Dict:
    model = MODELS.get(model_name)
    if model is None:
        return {"error": "Model not found"}
    return {"classes": list(model.names.values())}

@app.get("/health/")
async def health() -> Dict:
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
    }

@app.get("/version/")
async def version() -> Dict:
    return {
        "api_version": "1.0.0",
        "torch_version": torch.__version__,
        "models_available": list(MODELS.keys())
    }

@app.post("/predict-yolov8/")
async def predict_yolov8(
    files: List[UploadFile] = File(...)
) -> List[Dict]:
    results_data = []

    for file in files:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        try:
            rendered, preds, class_names = run_inference("yolov8", img)
        except Exception as e:
            return [{"filename": file.filename, "error": str(e)}]

        output_buffer = io.BytesIO()
        rendered.save(output_buffer, format="JPEG")
        base64_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        detections = []
        for pred in preds:
            x1, y1, x2, y2, conf, cls = pred
            detections.append({
                "class": class_names[int(cls)],
                "confidence": round(float(conf), 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

        results_data.append({
            "filename": file.filename,
            "model_used": "yolov8",
            "detections": detections,
            "image_base64": base64_image
        })

    return results_data
