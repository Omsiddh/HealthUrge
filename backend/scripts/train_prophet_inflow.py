import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.services.prophet_service import train_and_save_dummy_models

if __name__ == "__main__":
    # In a real scenario, this would only train inflow
    # For now, we reuse the shared function
    print("Training Inflow Model...")
    train_and_save_dummy_models()
