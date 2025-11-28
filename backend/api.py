from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import json
import os
from pathlib import Path
from disease_database import get_disease_info
import numpy as np

# Define Model Architecture (Must match train.py)
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b2'):
        super(PlantDiseaseModel, self).__init__()
        
        if model_name == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(weights='DEFAULT')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='DEFAULT')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CLASS_MAP_PATH = os.path.join(MODEL_DIR, "class_to_idx.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
model = None
class_to_idx = {}
idx_to_class = {}

def load_model():
    global model, class_to_idx, idx_to_class
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_MAP_PATH):
        print("Model or class mapping not found. Please train the model first.")
        return

    try:
        # Load class mapping
        with open(CLASS_MAP_PATH, 'r') as f:
            data = json.load(f)
            if 'class_to_idx' in data:
                class_to_idx = data['class_to_idx']
                model_name = data.get('model_name', 'efficientnet_b2')
            else:
                class_to_idx = data
                model_name = 'efficientnet_b2'
            
            idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Initialize model
        model = PlantDiseaseModel(len(class_to_idx), model_name=model_name)
        
        # Load weights
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(DEVICE)
        model.eval()
        print(f"Model loaded successfully: {model_name} with {len(class_to_idx)} classes")
        
    except Exception as e:
        print(f"Error loading model: {e}")

# Load model on startup
load_model()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((260, 260)), # Match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_valid_image(image: Image.Image) -> bool:
    """
    Simplified validation - accept all images and let the model decide.
    The confidence threshold will filter out truly invalid predictions.
    """
    # Just do basic checks
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Accept all images - let the model's confidence threshold handle filtering
    return True

@app.post("/predict")
async def predict(file: UploadFile = File(...), plant_type: str = None):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 1. Validate Image
        if not is_valid_image(image):
            return {
                "class": "invalid_image",
                "confidence": 0,
                "message": "The image does not appear to be a plant. Please upload a clear photo of a leaf or plant."
            }
            
        # 2. Preprocess
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        # 3. Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs, 5)
            
            top_predictions = []
            for i in range(5):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item() * 100
                class_name = idx_to_class[idx]
                
                top_predictions.append({
                    "class": class_name,
                    "confidence": round(prob, 2)
                })
        
        # 4. Low Confidence Rejection (Invalid Image Check)
        # If the model is not at least 20% confident in its top prediction, 
        # it's likely not a plant or a known disease.
        if top_predictions[0]['confidence'] < 20.0:
             return {
                "class": "invalid_image",
                "confidence": 0,
                "message": "The image content is unclear or does not appear to be a known plant. Please upload a clear photo of a plant leaf."
            }

        # 5. Filter by plant type if specified
        best_prediction = top_predictions[0]
        
        if plant_type:
            plant_type = plant_type.lower()
            filtered = [p for p in top_predictions if plant_type in p['class'].lower()]
            if filtered:
                best_prediction = filtered[0]
                # Adjust confidence if we filtered
                if best_prediction != top_predictions[0]:
                    best_prediction['confidence'] = min(best_prediction['confidence'] * 1.2, 99.9) # Boost slightly if it matches user intent
        
        # 5. Construct Response
        predicted_class = best_prediction['class']
        confidence = best_prediction['confidence']
        
        # Get disease info
        disease_info = get_disease_info(predicted_class)
        
        # Determine status message
        if confidence < 40:
            message = "⚠️ Low confidence. The model is unsure. Please ensure the image is clear and focused on the leaf."
        elif confidence < 70:
            message = "⚡ Moderate confidence. Verify the symptoms with the description below."
        else:
            message = "✅ High confidence prediction."

        return {
            "class": predicted_class,
            "confidence": confidence,
            "top3_predictions": top_predictions[:3],
            "message": message,
            "symptoms": disease_info.get("symptoms", []),
            "causes": disease_info.get("causes", []),
            "treatments": disease_info.get("treatments", {}),
            "prevention": disease_info.get("treatments", {}).get("prevention", [])
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
