from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd

class DeliveryCenter:
    """Represents a delivery center with efficiency and geographical characteristics."""
    def __init__(self, center_id, efficiency_factor, geographical_complexity):
        self.center_id = center_id
        self.efficiency_factor = efficiency_factor  # Range: 0.8 to 1.2
        self.geographical_complexity = geographical_complexity  # Range: 1 to 5
        
    def calculate_base_delivery_time(self, n_parcels):
        """Calculate base delivery time per parcel (in seconds)"""
        # Base time affected by center efficiency and geographical complexity
        base_time = 300  # 5 minutes base time per parcel
        # Economies of scale factor
        scale_factor = np.log(n_parcels) / np.log(200)  # normalized to 200 parcels
        # Adjust base time
        adjusted_time = (base_time * self.geographical_complexity * 
                        (1/self.efficiency_factor) * (1/scale_factor))
        return adjusted_time

class RoutingSystem:
    """Manages the routing system simulation including outage dates and delivery time calculations."""
    def __init__(self):
        self.outage_dates = self._generate_outage_dates()
        
    def _generate_outage_dates(self):
        """Generate 10 random outage dates for 2024"""
        all_dates = pd.date_range(start='2024-01-01', end='2024-12-31')
        return np.random.choice(all_dates, size=10, replace=False)
    
    def calculate_delivery_time(self, center, n_parcels, date, add_noise=True):
        """Calculate delivery time considering routing system status"""
        base_time = center.calculate_base_delivery_time(n_parcels)
        
        # Check if it's an outage day
        is_outage = date in self.outage_dates
        
        # Dynamic routing typically improves efficiency by 10-30%
        if not is_outage:
            improvement_factor = random.uniform(0.7, 0.9)
            delivery_time = base_time * improvement_factor
        else:
            # Manual routing might be less efficient or more variable
            manual_factor = random.uniform(0.9, 1.2)
            delivery_time = base_time * manual_factor
            
        # Add noise to simulate real-world variability
        if add_noise:
            noise_factor = np.random.normal(1, 0.1)
            delivery_time *= noise_factor
            
        return delivery_time, is_outage

def run_simulation(n_days=365, n_centers=41):
    """Run the complete simulation"""
    # Initialize systems
    routing_system = RoutingSystem()
    
    # Create delivery centers with random efficiency factors
    centers = [
        DeliveryCenter(
            i,
            efficiency_factor=random.uniform(0.8, 1.2),
            geographical_complexity=random.uniform(1, 5)
        ) for i in range(n_centers)
    ]
    
    # Prepare data collection
    results = []
    
    # Run simulation for each day and center
    dates = pd.date_range(start='2024-01-01', periods=n_days)
    
    for date in dates:
        for center in centers:
            # Generate random number of parcels (50-400)
            n_parcels = int(np.random.normal(200, 50))
            n_parcels = max(50, min(400, n_parcels))  # Clip to valid range
            
            # Calculate delivery time
            delivery_time, is_outage = routing_system.calculate_delivery_time(
                center, n_parcels, date
            )
            
            # Store results
            results.append({
                'date': date,
                'center_id': center.center_id,
                'n_parcels': n_parcels,
                'delivery_time_per_parcel': delivery_time,
                'is_outage': is_outage,
                'efficiency_factor': center.efficiency_factor,
                'geographical_complexity': center.geographical_complexity
            })
    
    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze simulation results"""
    # Basic statistics
    outage_stats = df.groupby('is_outage')['delivery_time_per_parcel'].agg(
        ['mean', 'std', 'count']
    )
    
    # Effect size calculation (Cohen's d)
    outage_data = df[df['is_outage']]['delivery_time_per_parcel']
    normal_data = df[~df['is_outage']]['delivery_time_per_parcel']
    
    pooled_std = np.sqrt((outage_data.std()**2 + normal_data.std()**2) / 2)
    effect_size = (outage_data.mean() - normal_data.mean()) / pooled_std
    
    return outage_stats, effect_size

if __name__ == "__main__":
    # Run simulation
    results_df = run_simulation()
    
    # Analyze results
    outage_stats, effect_size = analyze_results(results_df)
    
    print("\nDelivery Time Statistics (seconds per parcel):")
    print(outage_stats)
    print(f"\nEffect Size (Cohen's d): {effect_size:.3f}") 