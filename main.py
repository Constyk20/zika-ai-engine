# main.py - YOUR OWN TRAINED MODEL (DEPLOYS ON RENDER!)
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

# Load your models
clinical_model = joblib.load("models/ABSUTH_early_detection_model.pkl")
interpreter = tf.lite.Interpreter(model_path="models/malaria_lite.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("YOUR OWN MODELS LOADED!")

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.get("/")
def home():
    return {"status": "LIVE", "message": "Your own trained AI is ready!"}

@app.post("/predict")
def clinical(age: int = Form(...), sex: str = Form(...), travel: str = Form("No")):
    f = 1 if sex.upper() in ["F", "FEMALE"] else 0
    t = 1 if any(x in travel.lower() for x in ["yes", "lagos"]) else 0
    pred = clinical_model.predict([[age, f, t]])[0]
    prob = clinical_model.predict_proba([[age, f, t]])[0][1]
    return {"risk": "HIGH" if pred==1 else "LOW", "prob": round(float(prob), 4)}

@app.post("/detect-malaria")
async def malaria(file: UploadFile = File(...)):
    img = preprocess(await file.read())
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]["index"])[0][0]
    result = "Parasitized" if pred > 0.5 else "Uninfected"
    return {"result": result, "confidence": round(float(pred if pred>0.5 else 1-pred), 4)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
