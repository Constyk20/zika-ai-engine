# main.py - FIXED FOR NODE.JS INTEGRATION
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="ABSUTH Dual AI Engine", version="6.0")

# CORS - Allow your Node.js backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Node.js URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Loading AI models...")

# 1. Clinical Model (ABSUTH data)
try:
    clinical_model = joblib.load("models/ABSUTH_early_detection_model.pkl")
    print("✅ Clinical model loaded")
except Exception as e:
    print(f"❌ Clinical model error: {e}")
    clinical_model = None

# 2. Malaria TFLite Model
try:
    interpreter = tf.lite.Interpreter(model_path="models/malaria_lite.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Malaria TFLite model loaded")
except Exception as e:
    print(f"❌ Malaria model error: {e}")
    interpreter = None

print("🎯 AI Engine Ready!")

# Pydantic models for request validation
class PredictRequest(BaseModel):
    age: int
    sex: str
    travel_history: str

class HealthCheck(BaseModel):
    status: str
    clinical_model: str
    malaria_model: str
    endpoints: list

# Helper function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# Health check endpoint
@app.get("/", response_model=HealthCheck)
def health_check():
    return {
        "status": "ACTIVE",
        "clinical_model": "Ready" if clinical_model else "Not Loaded",
        "malaria_model": "Ready" if interpreter else "Not Loaded",
        "endpoints": ["/predict", "/detect-malaria", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "ABSUTH AI Engine"}

# Clinical Risk Prediction - MATCHES NODE.JS EXPECTATIONS
@app.post("/predict")
async def predict_clinical_risk(request: PredictRequest):
    """
    Predict Zika/Malaria risk from clinical features
    Expected by Node.js backend at /api/zika/predict
    """
    if not clinical_model:
        raise HTTPException(status_code=503, detail="Clinical model not loaded")
    
    try:
        # Parse input
        age = request.age
        sex = request.sex.strip().upper()
        travel_history = request.travel_history.lower()
        
        # Feature engineering (same as your training)
        is_female = 1 if sex in ["F", "FEMALE"] else 0
        has_travel = 1 if any(keyword in travel_history for keyword in 
                             ["yes", "lagos", "abuja", "port", "travel", "endemic", "y"]) else 0
        
        # Prepare features
        features = [[age, is_female, has_travel]]
        
        # Make prediction
        risk = clinical_model.predict(features)[0]
        probability = clinical_model.predict_proba(features)[0][1]
        
        # Format response to match Node.js expectations
        return {
            "success": True,
            "ai_prediction": {
                "risk_level": "HIGH - Urgent Testing Required" if risk == 1 else "LOW - Monitor Symptoms",
                "risk_probability": round(float(probability), 4),
                "risk_score": int(risk),
                "confidence": round(float(probability if risk == 1 else 1 - probability), 4),
                "recommendation": (
                    "🚨 URGENT: Refer for immediate Malaria & Zika lab tests. Consider ACT treatment." 
                    if risk == 1 else 
                    "✅ Low risk detected. Continue mosquito prevention and monitor symptoms."
                ),
                "model_source": "Random Forest trained on ABSUTH patient records (2025)"
            },
            "input_features": {
                "age": age,
                "sex": sex,
                "travel_history": request.travel_history,
                "processed_features": {
                    "is_female": is_female,
                    "has_travel_history": has_travel
                }
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Alternative endpoint that accepts Form data (for direct testing)
@app.post("/predict-form")
async def predict_clinical_form(
    age: int = Form(...),
    sex: str = Form(...),
    travel_history: str = Form("No")
):
    """Alternative endpoint for form data"""
    request = PredictRequest(age=age, sex=sex, travel_history=travel_history)
    return await predict_clinical_risk(request)

# Malaria Blood Smear Detection
@app.post("/detect-malaria")
async def detect_malaria(file: UploadFile = File(...)):
    """
    Detect malaria parasites from blood smear image
    """
    if not interpreter:
        raise HTTPException(status_code=503, detail="Malaria model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        img_array = preprocess_image(contents)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        # Format results
        probability = float(prediction)
        result = "Parasitized" if probability > 0.5 else "Uninfected"
        confidence = probability if result == "Parasitized" else 1 - probability
        
        return {
            "success": True,
            "malaria_detection": {
                "result": result,
                "confidence": round(confidence, 4),
                "parasite_probability": round(probability, 4),
                "recommendation": (
                    "🚨 URGENT: Malaria parasites detected! Start ACT treatment immediately and confirm with microscopy." 
                    if result == "Parasitized" else 
                    "✅ No malaria parasites detected in blood smear."
                ),
                "model_source": "CNN trained on NIH Malaria Dataset (27,550 images)"
            },
            "image_info": {
                "filename": file.filename,
                "size": len(contents)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Malaria detection error: {str(e)}")

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {
        "success": False,
        "error": str(exc),
        "message": "AI server encountered an error"
    }

# Run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)