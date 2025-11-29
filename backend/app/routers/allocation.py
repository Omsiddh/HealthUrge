import sys
import os
import math
from fastapi import APIRouter, HTTPException
from app.models.schemas import AllocationRequest, AllocationResponse, AllocationPlan
from app.services import gemini_service

# Add GNN directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../gnn"))

try:
    from service import GNNService
    gnn_service = GNNService(model_dir=os.path.join(os.path.dirname(__file__), "../../../gnn/models"))
except ImportError:
    print("GNN module not found or dependencies missing. GNN features will be limited.")
    gnn_service = None

router = APIRouter()

class GreedyAllocator:
    """
    Implements a Greedy algorithm for resource allocation.
    Strategy:
    1. Sort hospitals by stress level (Load / Capacity).
    2. Identify 'Source' (Overloaded) and 'Target' (Underutilized) hospitals.
    3. Iteratively move patients from highest stress source to lowest stress target
       until sources are relieved or targets are full.
    """
    @staticmethod
    def solve(hospitals):
        allocations = []
        
        # Deep copy to avoid mutating original state during simulation
        sim_hospitals = [h.copy() for h in hospitals]
        
        # Calculate initial load ratio
        for h in sim_hospitals:
            h['ratio'] = h['predicted_load'] / h['beds_capacity']
        
        # Sort: Highest ratio first (Sources), Lowest ratio last (Targets)
        sources = sorted([h for h in sim_hospitals if h['ratio'] > 1.0], key=lambda x: x['ratio'], reverse=True)
        targets = sorted([h for h in sim_hospitals if h['ratio'] < 0.8], key=lambda x: x['ratio']) # Only target those with < 80% load
        
        for source in sources:
            excess_patients = source['predicted_load'] - source['beds_capacity']
            
            for target in targets:
                if excess_patients <= 0:
                    break
                
                # Calculate available slots in target (aiming to fill up to 90%)
                available_slots = int(target['beds_capacity'] * 0.9) - target['predicted_load']
                
                if available_slots > 0:
                    # Determine transfer amount
                    transfer_count = min(excess_patients, available_slots)
                    
                    # Record allocation
                    allocations.append(AllocationPlan(
                        source_hospital_id=source['id'],
                        target_hospital_id=target['id'],
                        patient_count=transfer_count,
                        reason=f"Greedy: Relieving overload at {source['name']} (Load: {int(source['ratio']*100)}%)"
                    ))
                    
                    # Update simulation state
                    excess_patients -= transfer_count
                    target['predicted_load'] += transfer_count
                    
                    # Re-sort targets if needed (simplified here, we just continue)
        
        return allocations

@router.post("/generate", response_model=AllocationResponse)
def generate_allocation_plan(request: AllocationRequest):
    # 1. Fetch current hospital states (Mocking this part as we don't have live DB connection yet)
    hospitals_mock = [
        {"id": 1, "name": "KEM Hospital", "lat": 19.002, "lon": 72.842, "beds_capacity": 500, "predicted_load": 450},
        {"id": 2, "name": "Sion Hospital", "lat": 19.045, "lon": 72.865, "beds_capacity": 600, "predicted_load": 750}, # Highly Overloaded
        {"id": 3, "name": "Nair Hospital", "lat": 18.975, "lon": 72.825, "beds_capacity": 400, "predicted_load": 200}, # Underutilized
        {"id": 4, "name": "Cooper Hospital", "lat": 19.101, "lon": 72.837, "beds_capacity": 300, "predicted_load": 320}, # Slightly Overloaded
    ]

    # 2. Run GNN Analysis (Optional / Enrichment)
    stress_scores = {}
    if gnn_service:
        stress_scores = gnn_service.analyze_resources(hospitals_mock)
    
    # 3. Generate Allocation Plan using Greedy Algorithm
    allocations = GreedyAllocator.solve(hospitals_mock)

    # 4. Gemini Critique
    # Convert allocations to a cleaner JSON format for the LLM
    plan_summary = [
        {
            "from": a.source_hospital_id,
            "to": a.target_hospital_id,
            "count": a.patient_count,
            "reason": a.reason
        }
        for a in allocations
    ]
    
    critique = gemini_service.get_gemini_critique(plan_summary)
    
    return AllocationResponse(
        plan_id=f"plan-{request.date}",
        allocations=allocations,
        critique=critique,
        revised_allocations=None
    )
