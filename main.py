# main.py - FINAL DUAL MODE: Clinical + Malaria Blood Smear
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import joblib
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = FastAPI(
    title="ABSUTH Dual AI Engine",
    description="Clinical Risk (ABSUTH Data) + Malaria Blood Smear Detection",
    version="2.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load both models
print("Loading AI models...")
clinical_model = joblib.load("models/ABSUTH_early_detection_model.pkl")
malaria_model = tf.keras.models.load_model("models/malaria_cnn.h5")
print("Both models loaded: Clinical + Malaria CNN")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.get("/")
def home():
    return {"system": "ABSUTH Dual AI Active", "clinical": "Ready", "malaria_cnn": "Ready"}

# Clinical Prediction (your real ABSUTH model)
@app.post("/predict")
def predict_clinical(
    age: int = Form(...),
    sex: str = Form(...),
    travel_history: str = Form("No")
):
    is_female = 1 if sex.upper() in ["F", "FEMALE"] else 0
    has_travel = 1 if any(x in travel_history.lower() for x in ["yes", "lagos", "abuja"]) else 0
    features = [[age, is_female, has_travel]]
    risk = clinical_model.predict(features)[0]
    prob = clinical_model.predict_proba(features)[0][1]
    return {
        "risk_level": "HIGH" if risk == 1 else "LOW",
        "probability": round(float(prob), 4),
        "type": "clinical"
    }

# NEW: Malaria Detection from Blood Smear
@app.post("/detect-malaria")
async def detect_malaria(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_image(contents)
    pred = malaria_model.predict(img_array)[0][0]
    probability = float(pred)
    result = "Parasitized" if probability > 0.5 else "Uninfected"
    confidence = probability if result == "Parasitized" else 1 - probability

    return {
        "disease": "Malaria",
        "result": result,
        "confidence": round(confidence, 4),
        "probability_parasitized": round(probability, 4),
        "recommendation": "URGENT: Start ACT treatment" if result == "Parasitized" else "No malaria parasites",
        "model": "CNN trained on 27,550 NIH blood smears"
    }