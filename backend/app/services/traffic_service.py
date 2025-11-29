def get_traffic_delay(source_lat, source_lon, target_lat, target_lon):
    """
    Mock Traffic Service.
    Returns estimated delay in minutes based on distance and random traffic factor.
    """
    # Simple distance calculation (Euclidean for mock)
    dist = ((source_lat - target_lat)**2 + (source_lon - target_lon)**2)**0.5 * 111 # km
    
    # Base time: 2 mins per km
    base_time = dist * 2
    
    # Traffic factor: 1.5x to 3x delay
    import random
    traffic_factor = random.uniform(1.5, 3.0)
    
    return {
        "distance_km": round(dist, 1),
        "estimated_time_mins": round(base_time * traffic_factor),
        "traffic_level": "High" if traffic_factor > 2.5 else "Moderate"
    }
