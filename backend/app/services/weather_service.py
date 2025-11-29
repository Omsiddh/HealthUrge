import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_current_weather(lat: float, lon: float):
    """
    Fetch current weather from OpenWeatherMap.
    Falls back to mock data if API key is missing.
    """
    if not OPENWEATHER_API_KEY:
        return {
            "temp": 30.0,
            "humidity": 70,
            "condition": "Sunny (Mock)",
            "aqi": 150 # Mock AQI
        }
    
    try:
        # Weather
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url)
        data = res.json()
        
        # AQI (Separate endpoint usually, but mocking for simplicity if needed or using Air Pollution API)
        aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        aqi_res = requests.get(aqi_url)
        aqi_data = aqi_res.json()
        
        return {
            "temp": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "condition": data['weather'][0]['main'],
            "aqi": aqi_data['list'][0]['main']['aqi'] * 50 # Rough conversion to AQI scale
        }
    except Exception as e:
        print(f"Weather API Error: {e}")
        return {
            "temp": 28.0,
            "humidity": 65,
            "condition": "Unknown",
            "aqi": 100
        }
