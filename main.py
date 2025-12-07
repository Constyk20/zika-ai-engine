# main.py - FINAL DEPLOYABLE VERSION (No TensorFlow!)
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from imageai.Detection import ObjectDetection
import joblib
import os
import numpy as np

app = FastAPI(
    title="ABSUTH Dual AI Engine",
    description="Clinical Risk (Real ABSUTH Data) + Malaria Detection (Lightweight CNN)",
    version="3.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading models...")

# 1. Clinical Model (your real ABSUTH data)
clinical_model = joblib.load("models/ABSUTH_early_detection_model.pkl")

# 2. Lightweight Malaria Detector (NO TensorFlow!)
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("models/yolo-malaria.h5")
detector.loadModel()

print("Both models loaded: Clinical + Malaria (YOLO)")

@app.get("/")
def home():
    return {"status": "LIVE", "clinical": "Ready", "malaria": "Ready (YOLO)"}

@app.post("/predict")
def predict_clinical(age: int = Form(...), sex: str = Form(...), travel_history: str = Form("No")):
    is_female = 1 if sex.strip().upper() in ["F", "FEMALE"] else 0
    has_travel = 1 if any(x in travel_history.lower() for x in ["yes", "lagos", "abuja", "travel"]) else 0
    features = [[age, is_female, has_travel]]
    risk = clinical_model.predict(features)[0]
    prob = clinical_model.predict_proba(features)[0][1]
    
    return {
        "risk_level": "HIGH - Urgent Testing" if risk == 1 else "LOW - Monitor",
        "probability": round(float(prob), 4),
        "model": "Trained on ABSUTH Real Records"
    }

@app.post("/detect-malaria")
async def detect_malaria(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp_image.jpg", "wb") as f:
        f.write(contents)
    
    detections = detector.detectObjectsFromImage(
        input_image="temp_image.jpg",
        output_image_path="output.jpg",
        minimum_percentage_probability=30
    )
    
    parasites = [d for d in detections if d["name"] == "parasite"]
    result = "Parasitized" if parasites else "Uninfected"
    confidence = max([p["percentage_probability"] for p in parasites], default=0) / 100
    
    return {
        "result": result,
        "parasite_count": len(parasites),
        "confidence": round(confidence, 4),
        "recommendation": "URGENT: Start ACT" if result == "Parasitized" else "No parasites"
    }

# Required for Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)