from fastapi import APIRouter, HTTPException
from app.services import weather_service

router = APIRouter()

@router.get("/")
def get_city_weather(lat: float = 19.0760, lon: float = 72.8777):
    return weather_service.get_current_weather(lat, lon)
