from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import io
from PIL import Image
from pathlib import Path
from typing import List, Dict
import time
import base64
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

MODEL_PATH = Path(__file__).parent / "model" / "yolov5.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(MODEL_PATH))
model.eval()

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(Path(__file__).parent / "static" / "upload.html")

@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)) -> List[Dict]:
    results_data = []

    for file in files:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        results = model(img)
        results.render()

        output_img = Image.fromarray(results.ims[0])
        output_buffer = io.BytesIO()
        output_img.save(output_buffer, format='JPEG')
        base64_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

        detections = []
        for pred in results.pred[0]:
            x1, y1, x2, y2, conf, cls = pred.tolist()
            detections.append({
                "class": model.names[int(cls)],
                "confidence": round(float(conf), 4),
                "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]
            })

        results_data.append({
            "filename": file.filename,
            "detections": detections,
            "image_base64": base64_image
        })

    return results_data

@app.get("/classes/")
async def get_classes() -> Dict:
    return {"classes": list(model.names.values())}

@app.get("/health/")
async def health() -> Dict:
    try:
        return {
            "status": "ok",
            "model_loaded": True,
            "model_type": "YOLOv5",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/version/")
async def version() -> Dict:
    return {
        "api_version": "1.0.0",
        "torch_version": torch.__version__,
        "yolov5_version": "ultralytics/yolov5 latest" 
    }

@app.post("/predict-multiple/")
async def predict_multiple(files: List[UploadFile] = File(...)) -> List[Dict]:
    results_list = []
    for file in files:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        results = model(img)
        results.render()

        output_img_bytes = io.BytesIO()
        output_img = Image.fromarray(results.ims[0])
        output_img.save(output_img_bytes, format='JPEG')
        output_img_bytes.seek(0)
        base64_img = base64.b64encode(output_img_bytes.read()).decode("utf-8")

        detections = []
        for pred in results.pred[0]:
            x1, y1, x2, y2, conf, cls = pred.tolist()
            detections.append({
                "class": model.names[int(cls)],
                "confidence": round(float(conf), 4),
                "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]
            })

        results_list.append({
            "filename": file.filename,
            "detections": detections,
            "rendered_image_base64": base64_img
        })

    return results_list