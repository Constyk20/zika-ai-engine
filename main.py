# main.py - DUAL AI WITH FIXED TFLite (Deploy Ready)
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="ABSUTH Dual AI", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading models...")
clinical_model = joblib.load("models/ABSUTH_early_detection_model.pkl")

# Load your fixed TFLite model
interpreter = tf.lite.Interpreter(model_path="models/malaria_lite.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Models loaded!")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.get("/")
def home():
    return {"status": "LIVE", "clinical": "Ready", "malaria": "Ready"}

@app.post("/predict")
def predict_clinical(age: int = Form(...), sex: str = Form(...), travel_history: str = Form("No")):
    is_female = 1 if sex.strip().upper() in ["F", "FEMALE"] else 0
    has_travel = 1 if any(x in travel_history.lower() for x in ["yes", "lagos", "abuja"]) else 0
    features = [[age, is_female, has_travel]]
    risk = clinical_model.predict(features)[0]
    probability = clinical_model.predict_proba(features)[0][1]
    return {
        "risk_level": "HIGH" if risk == 1 else "LOW",
        "probability": round(float(probability), 4)
    }

@app.post("/detect-malaria")
async def detect_malaria(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_image(contents)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    result = "Parasitized" if prediction > 0.5 else "Uninfected"
    confidence = prediction if result == "Parasitized" else 1 - prediction
    return {
        "result": result,
        "confidence": round(float(confidence), 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))