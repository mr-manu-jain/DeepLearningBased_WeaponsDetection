from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import torch
import io
import base64
import os
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
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

# Define paths
MODEL_DIR = Path(__file__).parent / "model"
SAVE_DIR = Path(__file__).parent / "approved_predictions"

# Create the predictions directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Model loading block
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

def save_prediction_data(image_data: str, filename: str, model_name: str, detections: List[Dict], save_dir: Path):
    """
    Save the original image and detection data to disk
    
    Args:
        image_data: Base64 encoded image data
        filename: Original filename
        model_name: Name of the model used
        detections: List of detection dictionaries
        save_dir: Directory to save files
    """
    # Create a subfolder for images if not exists
    img_dir = save_dir / "images"
    os.makedirs(img_dir, exist_ok=True)
    # Save the original image if it doesn't already exist
    image_save_path = img_dir / filename

    if not image_save_path.exists():
        try:
            # Remove the prefix if it exists (e.g., "data:image/jpeg;base64,")
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]

            image_bytes = base64.b64decode(image_data)
            with open(image_save_path, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    # Save bounding box data
    txt_filename = f"{model_name}_{Path(filename).stem}.txt"
    txt_path = save_dir / txt_filename

    try:
        with open(txt_path, "w") as f:
            for det in detections:
                cls = det["class"]
                bbox = det["bbox"]
                # Format: imagename class x1 y1 x2 y2
                f.write(f"{filename} {cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        return True
    except Exception as e:
        print(f"Error saving detection data: {e}")
        return False


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(Path(__file__).parent / "static" / "upload.html")

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

                # Get original image in base64 for saving later if approved
                original_buffer = io.BytesIO()
                img.save(original_buffer, format="JPEG")
                original_base64 = base64.b64encode(original_buffer.getvalue()).decode("utf-8")

                detections = []
                for pred in preds:
                    x1, y1, x2, y2, conf, cls = pred
                    detection = {
                        "class": class_names[int(cls)],
                        "confidence": round(float(conf), 4),
                        "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]
                    }
                    detections.append(detection)

                per_model_results[model_name] = {
                    "detections": detections,
                    "image_base64": base64_image,
                    "original_base64": original_base64  # Add original image for saving later
                }

            except Exception as e:
                per_model_results[model_name] = {
                    "error": str(e),
                    "detections": [],
                    "image_base64": "",
                    "original_base64": ""
                }

        results_data.append({
            "filename": file.filename,
            "results": per_model_results
        })

    return results_data

@app.post("/save-approved/")
async def save_approved(approval_data: Dict[str, Any] = Body(...)) -> Dict:
    """
    Save approved detection data
    
    Expected body format:
    {
        "filename": "image.jpg",
        "model_name": "yolov5",
        "original_base64": "base64_encoded_image_data",
        "detections": [{"class": "knife", "bbox": [x1, y1, x2, y2], ...}, ...]
    }
    """
    try:
        filename = approval_data.get("filename")
        model_name = approval_data.get("model_name")
        original_base64 = approval_data.get("original_base64")
        detections = approval_data.get("detections", [])

        if not all([filename, model_name, original_base64]):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Missing required data"}
            )
        
        success = save_prediction_data(
            original_base64, 
            filename, 
            model_name, 
            detections, 
            SAVE_DIR
        )
        
        if success:
            return {"success": True, "message": f"Saved approved detection for {filename} using {model_name}"}
        else:
            return JSONResponse(status_code=500,content={"success": False, "error": "Failed to save data"})

    
    except Exception as e:
        return JSONResponse(status_code=500,content={"success": False, "error": str(e)})


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