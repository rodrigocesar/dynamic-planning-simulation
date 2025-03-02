"""Simulation module for routing outage analysis."""
import random
import numpy as np
import pandas as pd

from src.models.delivery_center import DeliveryCenter
from src.models.routing_system import RoutingSystem
from src.utils.config import Config


def run_simulation(n_days=365, n_centers=41):
    """Run the complete simulation"""
    # Initialize systems
    routing_system = RoutingSystem()
    
    # Create delivery centers
    centers = [DeliveryCenter(i) for i in range(n_centers)]
    
    # Prepare data collection
    results = []
    
    # Run simulation for each day and center
    dates = pd.date_range(start='2024-01-01', periods=n_days)
    
    for date in dates:
        for center in centers:
            # Generate tours for this day
            tours = center.generate_daily_tours()
            
            for tour_id, n_parcels in enumerate(tours):
                # Calculate delivery time
                delivery_time, is_outage = routing_system.calculate_delivery_time(
                    center, n_parcels, date
                )
                
                # Calculate time per parcel
                time_per_parcel = delivery_time / n_parcels
                
                # Store results
                results.append({
                    'date': date,
                    'center_id': center.center_id,
                    'tour_id': tour_id,
                    'n_parcels': n_parcels,
                    'delivery_time': delivery_time,
                    'delivery_time_per_parcel': time_per_parcel,
                    'is_outage': is_outage,
                })
    
    return pd.DataFrame(results) 