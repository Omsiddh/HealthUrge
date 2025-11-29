import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class HospitalDataGenerator:
    """Generate synthetic hospital load data for training the GNN"""
    
    def __init__(self, hospitals_csv_path):
        self.hospitals_df = pd.read_csv(hospitals_csv_path)
        self.hospital_profiles = self._create_hospital_profiles()
    
    def _create_hospital_profiles(self):
        """Create realistic load patterns for each hospital"""
        profiles = {}
        for _, hospital in self.hospitals_df.iterrows():
            # Different patterns based on hospital type
            if hospital['hospital_type'] == 'Government':
                base_load = 0.85  # Government hospitals typically very busy
                volatility = 0.15
            elif hospital['hospital_type'].startswith('Specialty'):
                base_load = 0.70  # Specialty hospitals more predictable
                volatility = 0.10
            else:  # Private
                base_load = 0.65  # Private hospitals moderate load
                volatility = 0.20
            
            # Urban density affects load
            if 'Dense' in hospital['area_type']:
                base_load += 0.10
            
            profiles[hospital['hospital_id']] = {
                'base_load': base_load,
                'volatility': volatility,
                'capacity': hospital['bed_capacity']
            }
        
        return profiles
    
    def generate_snapshot(self, timestamp):
        """Generate a single snapshot of all hospitals at a given time"""
        hospitals = []
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        for _, hospital in self.hospitals_df.iterrows():
            hid = hospital['hospital_id']
            profile = self.hospital_profiles[hid]
            
            # Time-based modulation
            hour_factor = 1.0
            if 8 <= hour <= 20:  # Day time - higher load
                hour_factor = 1.15
            elif 0 <= hour <= 6:  # Night - lower load
                hour_factor = 0.85
            
            # Weekend effect
            weekend_factor = 0.90 if day_of_week >= 5 else 1.0
            
            # Calculate load with randomness
            base_load = profile['base_load']
            volatility = profile['volatility']
            capacity = profile['capacity']
            
            load_ratio = base_load * hour_factor * weekend_factor
            load_ratio += np.random.normal(0, volatility)
            load_ratio = np.clip(load_ratio, 0.3, 1.5)  # 30% to 150% capacity
            
            predicted_load = int(load_ratio * capacity)
            
            hospitals.append({
                'id': hid,
                'beds_capacity': capacity,
                'predicted_load': predicted_load,
                'latitude': hospital['latitude'],
                'longitude': hospital['longitude'],
                'timestamp': timestamp.isoformat()
            })
        
        return hospitals
    
    def generate_historical_data(self, days=30, snapshots_per_day=24):
        """
        Generate historical data for training.
        
        Args:
            days: Number of days to simulate
            snapshots_per_day: How many snapshots per day (default: hourly)
        
        Returns:
            List of hospital snapshots
        """
        history = []
        start_date = datetime.now() - timedelta(days=days)
        
        total_snapshots = days * snapshots_per_day
        print(f"Generating {total_snapshots} snapshots over {days} days...")
        
        for day in range(days):
            for snapshot in range(snapshots_per_day):
                timestamp = start_date + timedelta(
                    days=day,
                    hours=snapshot * (24 / snapshots_per_day)
                )
                
                hospitals = self.generate_snapshot(timestamp)
                history.append(hospitals)
            
            if (day + 1) % 5 == 0:
                print(f"  Generated {day + 1}/{days} days...")
        
        print(f"✓ Generated {len(history)} snapshots")
        return history
    
    def save_training_data(self, history, output_path='training_data.json'):
        """Save generated data to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Saved training data to {output_path}")
    
    def generate_current_snapshot(self):
        """Generate a snapshot for current time (for testing inference)"""
        return self.generate_snapshot(datetime.now())


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = HospitalDataGenerator('hospitals.csv')
    
    # Generate 30 days of hourly data
    history = generator.generate_historical_data(days=30, snapshots_per_day=24)
    
    # Save to file
    generator.save_training_data(history, 'training_data.json')
    
    # Generate current snapshot for testing
    current = generator.generate_current_snapshot()
    print("\nCurrent snapshot sample:")
    print(f"Hospital {current[0]['id']}: {current[0]['predicted_load']}/{current[0]['beds_capacity']} beds")
