from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import forecast, allocation, advisory, admin, weather

app = FastAPI(title="MumbaiHacks Healthcare System API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast"])
app.include_router(allocation.router, prefix="/api/allocation", tags=["Allocation"])
app.include_router(advisory.router, prefix="/api/advisory", tags=["Advisory"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(weather.router, prefix="/api/weather", tags=["Weather"])

@app.get("/")
def read_root():
    return {"message": "MumbaiHacks API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}
