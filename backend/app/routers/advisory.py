from fastapi import APIRouter
from app.models.schemas import AdvisoryRequest, AdvisoryResponse
from app.services import gemini_service

router = APIRouter()

@router.post("/", response_model=AdvisoryResponse)
def get_advisory(request: AdvisoryRequest):
    advice_data = gemini_service.get_patient_advisory(
        request.symptoms, 
        {"lat": request.lat, "lon": request.lon}
    )
    
    # advice_data is now a dict (parsed JSON)
    return AdvisoryResponse(
        recommendation=advice_data.get("recommendation", "Consult Doctor"),
        reason=advice_data.get("reason", "Please consult a medical professional."),
        suggested_hospital=advice_data.get("nearest_hospital_type", "Local Clinic")
    )
