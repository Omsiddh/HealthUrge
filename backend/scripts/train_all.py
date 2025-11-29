import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.services.prophet_service import train_and_save_dummy_models

if __name__ == "__main__":
    print("Training Prophet models...")
    train_and_save_dummy_models()
    print("Done.")
