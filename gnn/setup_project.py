"""
Run this script to create all necessary files for the Hospital GNN project
Usage: python setup_project.py
"""

import os

# File contents as strings
FILES = {
    "data_generator.py": '''import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class HospitalDataGenerator:
    """Generate synthetic hospital load data for training the GNN"""
    
    def __init__(self, hospitals_csv_path):
        self.hospitals_df = pd.read_csv(hospitals_csv_path)
        self.hospital_profiles = self._create_hospital_profiles()
    
    def _create_hospital_profiles(self):
        """Create realistic load patterns for each hospital"""
        profiles = {}
        for _, hospital in self.hospitals_df.iterrows():
            # Different patterns based on hospital type
            if hospital['hospital_type'] == 'Government':
                base_load = 0.85  # Government hospitals typically very busy
                volatility = 0.15
            elif hospital['hospital_type'].startswith('Specialty'):
                base_load = 0.70  # Specialty hospitals more predictable
                volatility = 0.10
            else:  # Private
                base_load = 0.65  # Private hospitals moderate load
                volatility = 0.20
            
            # Urban density affects load
            if 'Dense' in hospital['area_type']:
                base_load += 0.10
            
            profiles[hospital['hospital_id']] = {
                'base_load': base_load,
                'volatility': volatility,
                'capacity': hospital['bed_capacity']
            }
        
        return profiles
    
    def generate_snapshot(self, timestamp):
        """Generate a single snapshot of all hospitals at a given time"""
        hospitals = []
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        for _, hospital in self.hospitals_df.iterrows():
            hid = hospital['hospital_id']
            profile = self.hospital_profiles[hid]
            
            # Time-based modulation
            hour_factor = 1.0
            if 8 <= hour <= 20:  # Day time - higher load
                hour_factor = 1.15
            elif 0 <= hour <= 6:  # Night - lower load
                hour_factor = 0.85
            
            # Weekend effect
            weekend_factor = 0.90 if day_of_week >= 5 else 1.0
            
            # Calculate load with randomness
            base_load = profile['base_load']
            volatility = profile['volatility']
            capacity = profile['capacity']
            
            load_ratio = base_load * hour_factor * weekend_factor
            load_ratio += np.random.normal(0, volatility)
            load_ratio = np.clip(load_ratio, 0.3, 1.5)  # 30% to 150% capacity
            
            predicted_load = int(load_ratio * capacity)
            
            hospitals.append({
                'id': hid,
                'beds_capacity': capacity,
                'predicted_load': predicted_load,
                'latitude': hospital['latitude'],
                'longitude': hospital['longitude'],
                'timestamp': timestamp.isoformat()
            })
        
        return hospitals
    
    def generate_historical_data(self, days=30, snapshots_per_day=24):
        """
        Generate historical data for training.
        
        Args:
            days: Number of days to simulate
            snapshots_per_day: How many snapshots per day (default: hourly)
        
        Returns:
            List of hospital snapshots
        """
        history = []
        start_date = datetime.now() - timedelta(days=days)
        
        total_snapshots = days * snapshots_per_day
        print(f"Generating {total_snapshots} snapshots over {days} days...")
        
        for day in range(days):
            for snapshot in range(snapshots_per_day):
                timestamp = start_date + timedelta(
                    days=day,
                    hours=snapshot * (24 / snapshots_per_day)
                )
                
                hospitals = self.generate_snapshot(timestamp)
                history.append(hospitals)
            
            if (day + 1) % 5 == 0:
                print(f"  Generated {day + 1}/{days} days...")
        
        print(f"✓ Generated {len(history)} snapshots")
        return history
    
    def save_training_data(self, history, output_path='training_data.json'):
        """Save generated data to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Saved training data to {output_path}")
    
    def generate_current_snapshot(self):
        """Generate a snapshot for current time (for testing inference)"""
        return self.generate_snapshot(datetime.now())


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = HospitalDataGenerator('hospitals.csv')
    
    # Generate 30 days of hourly data
    history = generator.generate_historical_data(days=30, snapshots_per_day=24)
    
    # Save to file
    generator.save_training_data(history, 'training_data.json')
    
    # Generate current snapshot for testing
    current = generator.generate_current_snapshot()
    print("\\nCurrent snapshot sample:")
    print(f"Hospital {current[0]['id']}: {current[0]['predicted_load']}/{current[0]['beds_capacity']} beds")
''',

    "train_gnn.py": '''"""
Complete training script for the Hospital Resource GNN
Usage: python train_gnn.py
"""

import json
import sys
from service import GNNService
from data_generator import HospitalDataGenerator

def main():
    print("\\n" + "="*70)
    print("Hospital Resource Management - GNN Training Pipeline")
    print("="*70 + "\\n")
    
    # Step 1: Generate training data
    print("STEP 1: Generating Training Data")
    print("-" * 70)
    
    generator = HospitalDataGenerator('hospitals.csv')
    
    # Generate 30 days of hourly snapshots (720 samples)
    history = generator.generate_historical_data(
        days=30, 
        snapshots_per_day=24
    )
    
    # Save for future use
    generator.save_training_data(history, 'training_data.json')
    
    print(f"\\n✓ Generated {len(history)} training samples\\n")
    
    # Step 2: Initialize GNN Service
    print("STEP 2: Initializing GNN Service")
    print("-" * 70)
    
    gnn_service = GNNService(model_dir="models")
    print()
    
    # Step 3: Train the model
    print("STEP 3: Training GNN Model")
    print("-" * 70)
    
    training_results = gnn_service.train(
        hospital_history=history,
        epochs=100,
        learning_rate=0.001,
        validation_split=0.2,
        early_stopping_patience=15
    )
    
    if training_results:
        print("\\nTraining Results:")
        print(f"  Epochs Trained: {training_results['epochs_trained']}")
        print(f"  Best Val Loss: {training_results['best_val_loss']:.4f}")
        
        if training_results['final_metrics']:
            metrics = training_results['final_metrics']
            print(f"\\nFinal Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
    
    # Step 4: Test inference
    print("\\n" + "="*70)
    print("STEP 4: Testing Inference")
    print("-" * 70)
    
    # Generate current snapshot
    current_snapshot = generator.generate_current_snapshot()
    
    # Run inference
    predictions = gnn_service.analyze_resources(current_snapshot)
    
    print("\\nStress Predictions for Current Snapshot:")
    print("-" * 70)
    
    for hospital_id, pred in sorted(predictions.items(), 
                                     key=lambda x: x[1]['stress_score'], 
                                     reverse=True):
        stress = pred['stress_score']
        level = pred['stress_level']
        load = pred['current_load']
        capacity = pred['capacity']
        ratio = pred['load_ratio']
        
        # Color coding for terminal
        if level == 'critical':
            marker = '🔴'
        elif level == 'high':
            marker = '🟠'
        elif level == 'moderate':
            marker = '🟡'
        else:
            marker = '🟢'
        
        print(f"{marker} {hospital_id}: Stress={stress:.3f} ({level:8s}) | "
              f"Load={load:4d}/{capacity:4d} ({ratio:.1%})")
    
    # Step 5: Save summary
    print("\\n" + "="*70)
    print("STEP 5: Saving Training Summary")
    print("-" * 70)
    
    summary = gnn_service.get_training_summary()
    
    with open('models/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Training summary saved to models/training_summary.json")
    
    print("\\n" + "="*70)
    print("✓ Training Pipeline Complete!")
    print("="*70)
    print("\\nNext Steps:")
    print("  1. Review training metrics in models/training_history.json")
    print("  2. Test the API server: python api_server.py")
    print("  3. Monitor predictions and retrain as needed")
    print("\\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
''',

    "api_server.py": '''"""
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
'''
}

def create_files():
    """Create all project files"""
    print("\n" + "="*70)
    print("Hospital GNN Project Setup")
    print("="*70 + "\n")
    
    created = []
    skipped = []
    
    for filename, content in FILES.items():
        if os.path.exists(filename):
            response = input(f"⚠️  {filename} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                skipped.append(filename)
                continue
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        created.append(filename)
        print(f"✓ Created {filename}")
    
    # Create models directory
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✓ Created models/ directory")
    
    print("\n" + "="*70)
    print("Setup Complete!")
    print("="*70)
    
    if created:
        print("\n✓ Created files:")
        for f in created:
            print(f"  - {f}")
    
    if skipped:
        print("\n⊘ Skipped files:")
        for f in skipped:
            print(f"  - {f}")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("\n1. Install dependencies:")
    print("   pip install torch torch-geometric scikit-learn pandas numpy fastapi uvicorn")
    print("\n2. Train your model:")
    print("   python train_gnn.py")
    print("\n3. Start the API server:")
    print("   uvicorn api_server:app --reload")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        create_files()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()