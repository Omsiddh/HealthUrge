from fastapi import APIRouter, HTTPException
from app.models.schemas import ForecastResponse
from app.services import prophet_service

router = APIRouter()

@router.get("/", response_model=ForecastResponse)
def get_forecast(hospital_id: int, days: int = 7):
    # Fetch Inflow
    inflow_data = prophet_service.get_forecast(hospital_id, days, "inflow")
    
    # Fetch Capacity
    capacity_data = prophet_service.get_forecast(hospital_id, days, "capacity")
    
    return ForecastResponse(
        hospital_id=hospital_id,
        dates=inflow_data["dates"],
        predicted_inflow=inflow_data["values"],
        predicted_capacity=capacity_data["values"]
    )
