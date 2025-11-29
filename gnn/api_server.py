"""
FastAPI server for Hospital Resource GNN
Usage: uvicorn api_server:app --reload
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

from service import GNNService
from data_generator import HospitalDataGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Resource Management API",
    description="GNN-based hospital stress prediction and resource allocation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
gnn_service = GNNService(model_dir="models")
generator = HospitalDataGenerator('hospitals.csv')

# Pydantic models
class Hospital(BaseModel):
    id: str
    beds_capacity: int
    predicted_load: int
    latitude: float
    longitude: float

class PredictionRequest(BaseModel):
    hospitals: List[Hospital]

class PredictionResponse(BaseModel):
    hospital_id: str
    stress_score: float
    stress_level: str
    current_load: int
    capacity: int
    load_ratio: float
    recommendation: str

class TrainingRequest(BaseModel):
    days: int = Field(default=30, ge=7, le=90, description="Days of data to generate")
    epochs: int = Field(default=100, ge=10, le=500, description="Training epochs")
    learning_rate: float = Field(default=0.001, gt=0, lt=1)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


# Routes
@app.get("/", response_model=Dict)
async def root():
    """API information"""
    return {
        "name": "Hospital Resource Management API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_current": "/predict/current",
            "train": "/train",
            "metrics": "/metrics",
            "hospitals": "/hospitals"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health"""
    return HealthResponse(
        status="healthy",
        model_loaded=gnn_service.model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/hospitals", response_model=List[Dict])
async def get_hospitals():
    """Get list of all hospitals"""
    hospitals_df = pd.read_csv('hospitals.csv')
    return hospitals_df.to_dict('records')

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_stress(request: PredictionRequest):
    """
    Predict stress levels for given hospital states
    """
    if not gnn_service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to dict format
    hospitals = [h.dict() for h in request.hospitals]
    
    # Run inference
    predictions = gnn_service.analyze_resources(hospitals)
    
    if not predictions:
        raise HTTPException(status_code=500, detail="Prediction failed")
    
    # Format response with recommendations
    results = []
    for hospital_id, pred in predictions.items():
        # Generate recommendation
        stress = pred['stress_score']
        level = pred['stress_level']
        
        if level == 'critical':
            recommendation = "URGENT: Divert incoming patients, activate overflow protocols"
        elif level == 'high':
            recommendation = "Consider diverting non-critical cases, prepare surge capacity"
        elif level == 'moderate':
            recommendation = "Monitor closely, prepare to scale resources"
        else:
            recommendation = "Normal operations, capacity available"
        
        results.append(PredictionResponse(
            hospital_id=hospital_id,
            stress_score=pred['stress_score'],
            stress_level=pred['stress_level'],
            current_load=pred['current_load'],
            capacity=pred['capacity'],
            load_ratio=pred['load_ratio'],
            recommendation=recommendation
        ))
    
    # Sort by stress score (highest first)
    results.sort(key=lambda x: x.stress_score, reverse=True)
    
    return results

@app.get("/predict/current", response_model=List[PredictionResponse])
async def predict_current():
    """
    Predict stress for current hospital snapshot (synthetic data for demo)
    """
    # Generate current snapshot
    current_snapshot = generator.generate_current_snapshot()
    
    # Create request
    request = PredictionRequest(
        hospitals=[Hospital(**h) for h in current_snapshot]
    )
    
    return await predict_stress(request)

@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train the GNN model (runs in background)
    """
    def train_task():
        try:
            # Generate training data
            history = generator.generate_historical_data(
                days=request.days,
                snapshots_per_day=24
            )
            
            # Train model
            results = gnn_service.train(
                hospital_history=history,
                epochs=request.epochs,
                learning_rate=request.learning_rate,
                validation_split=0.2,
                early_stopping_patience=15
            )
            
            print(f"✓ Training complete: {results}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
    
    background_tasks.add_task(train_task)
    
    return {
        "status": "training_started",
        "message": f"Training initiated with {request.days} days of data for {request.epochs} epochs",
        "note": "Training runs in background. Check /metrics for progress."
    }

@app.get("/metrics", response_model=Dict)
async def get_metrics():
    """
    Get current model training metrics
    """
    if not gnn_service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    summary = gnn_service.get_training_summary()
    
    if summary == "No training history available":
        return {"status": "no_history", "message": summary}
    
    return {
        "status": "available",
        "summary": summary,
        "model_path": "models/gnn_resource_model.pth"
    }

@app.get("/recommendations/{hospital_id}")
async def get_hospital_recommendation(hospital_id: str):
    """
    Get detailed recommendation for a specific hospital
    """
    # Generate current snapshot
    current_snapshot = generator.generate_current_snapshot()
    
    # Run inference
    predictions = gnn_service.analyze_resources(current_snapshot)
    
    if hospital_id not in predictions:
        raise HTTPException(status_code=404, detail=f"Hospital {hospital_id} not found")
    
    pred = predictions[hospital_id]
    stress = pred['stress_score']
    level = pred['stress_level']
    
    # Get hospital info
    hospitals_df = pd.read_csv('hospitals.csv')
    hospital_info = hospitals_df[hospitals_df['hospital_id'] == hospital_id].to_dict('records')[0]
    
    # Detailed recommendations based on stress level
    actions = []
    if level == 'critical':
        actions = [
            "Immediately activate disaster response protocol",
            "Divert all non-critical incoming ambulances",
            "Request additional staff from neighboring facilities",
            "Activate overflow wards and temporary beds",
            "Coordinate with district health authority"
        ]
    elif level == 'high':
        actions = [
            "Alert senior management team",
            "Consider diverting non-emergency cases",
            "Prepare surge capacity plans",
            "Contact nearby hospitals for potential transfers",
            "Monitor bed availability every 30 minutes"
        ]
    elif level == 'moderate':
        actions = [
            "Continue monitoring bed occupancy",
            "Expedite discharge planning for stable patients",
            "Ensure adequate staff coverage for next shift",
            "Review elective procedure schedule"
        ]
    else:
        actions = [
            "Maintain normal operations",
            "Standard monitoring protocols",
            "Capacity available for incoming patients"
        ]
    
    return {
        "hospital_id": hospital_id,
        "hospital_name": hospital_info['hospital_name'],
        "current_status": {
            "stress_score": stress,
            "stress_level": level,
            "current_load": pred['current_load'],
            "capacity": pred['capacity'],
            "load_ratio": pred['load_ratio'],
            "available_beds": pred['capacity'] - pred['current_load']
        },
        "recommended_actions": actions,
        "nearby_hospitals": _get_nearby_hospitals(hospital_info, predictions),
        "timestamp": datetime.now().isoformat()
    }

def _get_nearby_hospitals(hospital_info, all_predictions):
    """Find nearby hospitals with lower stress"""
    from graph_utils import calculate_distance
    
    hospitals_df = pd.read_csv('hospitals.csv')
    nearby = []
    
    for _, other in hospitals_df.iterrows():
        if other['hospital_id'] == hospital_info['hospital_id']:
            continue
        
        dist = calculate_distance(
            hospital_info['latitude'], hospital_info['longitude'],
            other['latitude'], other['longitude']
        )
        
        if dist <= 20:  # Within 20km
            other_pred = all_predictions.get(other['hospital_id'], {})
            nearby.append({
                'hospital_id': other['hospital_id'],
                'hospital_name': other['hospital_name'],
                'distance_km': round(dist, 2),
                'stress_level': other_pred.get('stress_level', 'unknown'),
                'available_beds': other_pred.get('capacity', 0) - other_pred.get('current_load', 0)
            })
    
    # Sort by stress level (low first) and distance
    nearby.sort(key=lambda x: (x['stress_level'], x['distance_km']))
    
    return nearby[:5]  # Return top 5


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
