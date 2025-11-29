from pydantic import BaseModel
from typing import List, Optional, Dict

class ForecastRequest(BaseModel):
    hospital_id: int
    days: int = 7

class ForecastResponse(BaseModel):
    hospital_id: int
    dates: List[str]
    predicted_inflow: List[float]
    predicted_capacity: List[float]

class AllocationRequest(BaseModel):
    date: str
    
class AllocationPlan(BaseModel):
    source_hospital_id: int
    target_hospital_id: int
    patient_count: int
    reason: str

class AllocationResponse(BaseModel):
    plan_id: str
    allocations: List[AllocationPlan]
    critique: str
    revised_allocations: Optional[List[AllocationPlan]] = None

class AdvisoryRequest(BaseModel):
    symptoms: str
    lat: float
    lon: float
    severity: str = "Medium" # Low, Medium, High

class AdvisoryResponse(BaseModel):
    recommendation: str
    reason: str
    suggested_hospital: Optional[str] = None

class HospitalCreate(BaseModel):
    name: str
    lat: float
    lon: float
    beds_capacity: int

class SurgeRequest(BaseModel):
    hospital_id: int
    factor: float # e.g. 1.2 for 20% increase
