# main.py - OPTIMIZED DUAL AI ENGINE (Clinical + Malaria) for Render.com
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI(
    title="ABSUTH Dual AI Engine",
    description="Clinical Risk Prediction (Real ABSUTH Data) + Malaria Parasite Detection (CNN)",
    version="2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
print("Loading AI models...")

# 1. Clinical Model (ABSUTH-trained Random Forest)
clinical_model = joblib.load("models/ABSUTH_early_detection_model.pkl")

# 2. Malaria CNN (TensorFlow Lite - lightweight!)
try:
    import tensorflow as tf
    malaria_interpreter = tf.lite.Interpreter(model_path="models/malaria_cnn.tflite")
    malaria_interpreter.allocate_tensors()
    input_details = malaria_interpreter.get_input_details()
    output_details = malaria_interpreter.get_output_details()
    print("✅ Using TensorFlow Lite (optimized)")
except ImportError:
    # Fallback: Try tflite_runtime (smaller package)
    import tflite_runtime.interpreter as tflite
    malaria_interpreter = tflite.Interpreter(model_path="models/malaria_cnn.tflite")
    malaria_interpreter.allocate_tensors()
    input_details = malaria_interpreter.get_input_details()
    output_details = malaria_interpreter.get_output_details()
    print("✅ Using tflite_runtime (ultra-lightweight)")

print("Both models loaded successfully!")
print("Clinical Model: ABSUTH Real Patient Records")
print("Malaria CNN: Trained on 27,550 NIH blood smear images (TFLite)")

# Preprocess image for CNN
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# Health check
@app.get("/")
def home():
    return {
        "system": "ABSUTH Dual AI Engine",
        "status": "ACTIVE",
        "clinical_model": "Ready (Real ABSUTH Data)",
        "malaria_cnn": "Ready (27,550 NIH Images - TFLite)",
        "endpoints": ["/predict", "/detect-malaria"],
        "optimization": "TensorFlow Lite for fast deployment"
    }

# 1. Clinical Risk Prediction (Age, Sex, Travel History)
@app.post("/predict")
def predict_clinical(
    age: int = Form(..., description="Patient age"),
    sex: str = Form(..., description="M or F"),
    travel_history: str = Form("No", description="Any recent travel?")
):
    is_female = 1 if sex.strip().upper() in ["F", "FEMALE"] else 0
    has_travel = 1 if any(x in travel_history.lower() for x in ["yes", "lagos", "abuja", "port", "travel", "endemic"]) else 0
    
    features = [[age, is_female, has_travel]]
    risk = clinical_model.predict(features)[0]
    probability = clinical_model.predict_proba(features)[0][1]

    return {
        "ai_prediction": {
            "risk_level": "HIGH - Urgent Testing Required" if risk == 1 else "LOW - Monitor Symptoms",
            "risk_probability": round(float(probability), 4),
            "recommendation": "Refer for Malaria & Zika lab tests immediately" if risk == 1 
                            else "Continue mosquito prevention and monitoring",
            "model_source": "Random Forest trained on real ABSUTH patient records (2025)"
        },
        "input_features": {"age": age, "sex": sex, "travel_history": travel_history}
    }

# 2. Malaria Parasite Detection from Blood Smear Image (TFLite)
@app.post("/detect-malaria")
async def detect_malaria(file: UploadFile = File(..., description="Upload blood smear image")):
    contents = await file.read()
    img_array = preprocess_image(contents)
    
    # Run inference with TFLite
    malaria_interpreter.set_tensor(input_details[0]['index'], img_array)
    malaria_interpreter.invoke()
    prediction = malaria_interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    probability = float(prediction)
    result = "Parasitized" if probability > 0.5 else "Uninfected"
    confidence = probability if result == "Parasitized" else 1 - probability

    return {
        "malaria_detection": {
            "result": result,
            "confidence": round(confidence, 4),
            "parasite_probability": round(probability, 4),
            "recommendation": "URGENT: Start ACT treatment + Confirm with microscopy" if result == "Parasitized"
                            else "No malaria parasites detected",
            "model_source": "Convolutional Neural Network trained on NIH Malaria Dataset (27,550 images)"
        },
        "image_received": file.filename
    }

# Required for Render.com deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)