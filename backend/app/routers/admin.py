from fastapi import APIRouter, HTTPException
from app.models.schemas import HospitalCreate, SurgeRequest
from app.services import prophet_service

router = APIRouter()

# Mock database for hospitals (In-memory for hackathon)
HOSPITALS_DB = [
    {"id": 1, "name": "KEM Hospital", "lat": 19.002, "lon": 72.842, "beds_capacity": 500},
    {"id": 2, "name": "Sion Hospital", "lat": 19.045, "lon": 72.865, "beds_capacity": 600},
    {"id": 3, "name": "Nair Hospital", "lat": 18.975, "lon": 72.825, "beds_capacity": 400},
    {"id": 4, "name": "Cooper Hospital", "lat": 19.101, "lon": 72.837, "beds_capacity": 300},
]

@router.post("/outbreak-predict")
def predict_outbreak():
    return {
        "level": "Low",
        "reason": "AQI is stable and no major festivals this week."
    }

@router.post("/add-hospital")
def add_hospital(hospital: HospitalCreate):
    new_id = len(HOSPITALS_DB) + 1
    new_hospital = {
        "id": new_id,
        "name": hospital.name,
        "lat": hospital.lat,
        "lon": hospital.lon,
        "beds_capacity": hospital.beds_capacity
    }
    HOSPITALS_DB.append(new_hospital)
    # Train initial model for new hospital
    prophet_service.train_model(new_id)
    return {"status": "success", "hospital": new_hospital}

@router.get("/hospitals")
def get_hospitals():
    return HOSPITALS_DB

@router.post("/simulate-surge")
def simulate_surge(request: SurgeRequest):
    return prophet_service.simulate_surge(request.hospital_id, request.factor)

@router.post("/retrain/{hospital_id}")
def retrain_hospital_model(hospital_id: int):
    return prophet_service.retrain_model(hospital_id)
