import os
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
from datetime import timedelta, date
import logging

# Suppress Prophet logs
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../data")
INFLOW_DATA_PATH = os.path.join(MODEL_DIR, "inflow_data.csv")
CAPACITY_DATA_PATH = os.path.join(MODEL_DIR, "capacity_data.csv")

# Global state for surge simulation (in-memory for hackathon)
SURGE_STATE = {}

def get_model_path(hospital_id: int, type: str):
    return os.path.join(MODEL_DIR, f"prophet_{type}_{hospital_id}.joblib")

def generate_synthetic_data(hospital_id: int, days=365, type="inflow"):
    dates = pd.date_range(start=date.today() - timedelta(days=days), periods=days)
    
    # Use hospital_id as seed for uniqueness
    np.random.seed(hospital_id)
    
    if type == "inflow":
        # Unique pattern per hospital
        base = 100 + (hospital_id * 10) # Different base
        trend = np.linspace(0, 50 + (hospital_id * 2), days)
        # Phase shift seasonality based on hospital_id
        seasonality = 20 * np.sin(2 * np.pi * (dates.dayofweek + hospital_id) / 7)
        noise = np.random.normal(0, 10, days)
        y = base + trend + seasonality + noise
    else:
        # Capacity
        base = 500 + (hospital_id * 20)
        trend = np.linspace(0, -10, days) 
        seasonality = 5 * np.sin(2 * np.pi * dates.dayofweek / 7)
        noise = np.random.normal(0, 5, days)
        y = base + trend + seasonality + noise

    return pd.DataFrame({'ds': dates, 'y': y})

def generate_fallback_forecast(hospital_id: int, days: int, type: str):
    """
    Generates a synthetic forecast if Prophet fails (e.g. on Windows due to cmdstanpy issues).
    """
    future_dates = pd.date_range(start=date.today(), periods=days)
    
    # Use same logic as synthetic data generation but for future dates
    np.random.seed(hospital_id + 1000) # Different seed for future
    
    if type == "inflow":
        base = 100 + (hospital_id * 10) + 50 + (hospital_id * 2) # End of trend
        trend = np.linspace(0, 5, days) # Slight upward trend
        seasonality = 20 * np.sin(2 * np.pi * (future_dates.dayofweek + hospital_id) / 7)
        noise = np.random.normal(0, 10, days)
        y = base + trend + seasonality + noise
    else:
        base = 500 + (hospital_id * 20) - 10 # End of trend
        trend = np.linspace(0, -1, days) 
        seasonality = 5 * np.sin(2 * np.pi * future_dates.dayofweek / 7)
        noise = np.random.normal(0, 5, days)
        y = base + trend + seasonality + noise
        
    return {
        "dates": future_dates.strftime("%Y-%m-%d").tolist(),
        "values": y.tolist()
    }

def train_model(hospital_id: int):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    try:
        # Inflow
        df_inflow = generate_synthetic_data(hospital_id, type="inflow")
        m_inflow = Prophet(daily_seasonality=True)
        m_inflow.fit(df_inflow)
        joblib.dump(m_inflow, get_model_path(hospital_id, "inflow"))
        
        # Capacity
        df_cap = generate_synthetic_data(hospital_id, type="capacity")
        m_cap = Prophet(daily_seasonality=True)
        m_cap.fit(df_cap)
        joblib.dump(m_cap, get_model_path(hospital_id, "capacity"))
        return True
    except Exception as e:
        print(f"Prophet training failed for Hospital {hospital_id}: {e}")
        return False

def get_forecast(hospital_id: int, days: int, type: str):
    model_path = get_model_path(hospital_id, type)
    
    model = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except:
            pass
            
    if model is None:
        success = train_model(hospital_id)
        if success:
            try:
                model = joblib.load(model_path)
            except:
                pass
    
    # If model is still None or training failed, use fallback
    if model is None:
        print(f"Using fallback forecast for Hospital {hospital_id} ({type})")
        data = generate_fallback_forecast(hospital_id, days, type)
        predictions = np.array(data["values"])
        dates = data["dates"]
    else:
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        future_forecast = forecast.tail(days)
        predictions = future_forecast['yhat'].values
        dates = future_forecast['ds'].dt.strftime("%Y-%m-%d").tolist()
    
    # Apply Surge Simulation if active
    if type == "inflow" and hospital_id in SURGE_STATE:
        factor = SURGE_STATE[hospital_id]
        predictions = predictions * factor
    
    return {
        "dates": dates,
        "values": predictions.tolist()
    }

def simulate_surge(hospital_id: int, factor: float):
    SURGE_STATE[hospital_id] = factor
    return {"status": "success", "hospital_id": hospital_id, "surge_factor": factor}

def retrain_model(hospital_id: int):
    success = train_model(hospital_id)
    if success:
        return {"status": "success", "message": f"Prophet Models retrained for Hospital {hospital_id}"}
    else:
        return {"status": "warning", "message": f"Prophet training failed. Using fallback logic for Hospital {hospital_id}"}
