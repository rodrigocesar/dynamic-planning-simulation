"""Module containing the RoutingSystem class implementation."""
import random
import numpy as np
import pandas as pd
from src.utils.config import Config


class RoutingSystem:
    """Manages the routing system simulation including outage dates and delivery time."""
    
    def __init__(self):
        self.config = Config()
        self.outage_dates = self._generate_outage_dates()
        
    def _generate_outage_dates(self):
        """Generate random outage dates for 2024."""
        all_dates = pd.date_range(start='2024-01-01', end='2024-12-31')
        return np.random.choice(all_dates, size=self.config.N_OUTAGES, 
                                replace=False)
    
    def calculate_delivery_time(self, center, n_parcels, date, add_noise=True):
        """Calculate total delivery time for tour in seconds."""
        base_time = center.calculate_base_delivery_time(n_parcels)
        
        # Check if it's an outage day
        is_outage = date in self.outage_dates
        
        if not is_outage:
            # Normal operation with dynamic routing
            improvement_factor = random.uniform(
                self.config.DYNAMIC_ROUTING_MIN_FACTOR,
                self.config.DYNAMIC_ROUTING_MAX_FACTOR
            )
            delivery_time = base_time * improvement_factor
        else:
            # During outages, efficiency depends on parcel count
            # Define the crossover point where manual routing becomes less efficient
            crossover_point = self.config.MANUAL_ROUTING_CROSSOVER_POINT
            
            # Calculate efficiency factor based on parcel count
            # For small parcel counts: more efficient (lower factor)
            # For large parcel counts: less efficient (higher factor)
            if n_parcels < crossover_point:
                # Small parcel counts - manual routing is more efficient
                # Linear interpolation from best efficiency to neutral
                efficiency_range = self.config.MANUAL_ROUTING_SMALL_MAX_FACTOR - self.config.MANUAL_ROUTING_SMALL_MIN_FACTOR
                position = n_parcels / crossover_point  # 0 to 1
                manual_factor = self.config.MANUAL_ROUTING_SMALL_MIN_FACTOR + (efficiency_range * position)
            else:
                # Large parcel counts - manual routing is less efficient
                # Linear interpolation from neutral to worst efficiency
                max_parcels = self.config.MAX_PARCELS_PER_TOUR
                position = min(1.0, (n_parcels - crossover_point) / (max_parcels - crossover_point))  # 0 to 1
                efficiency_range = self.config.MANUAL_ROUTING_LARGE_MAX_FACTOR - self.config.MANUAL_ROUTING_LARGE_MIN_FACTOR
                manual_factor = self.config.MANUAL_ROUTING_LARGE_MIN_FACTOR + (efficiency_range * position)
            
            delivery_time = base_time * manual_factor
            
        if add_noise:
            # U-shaped variance pattern:
            # - Higher variance for low parcel counts (< 100)
            # - Lower variance for medium parcel counts (100-300)
            # - Higher variance for high parcel counts (> 300)
            
            # Calculate base noise level
            base_noise_std = self.config.NOISE_FACTOR_STD
            
            # Adjust noise based on parcel count - U-shaped function
            min_parcels = self.config.MIN_PARCELS_PER_TOUR
            max_parcels = self.config.MAX_PARCELS_PER_TOUR
            mid_point = (min_parcels + max_parcels) / 2
            
            # Calculate distance from midpoint (normalized to 0-1 range)
            distance_from_mid = abs(n_parcels - mid_point) / (max_parcels - min_parcels) * 2
            
            # Apply U-shaped variance multiplier (1.0 at midpoint, up to 3.0 at extremes)
            variance_multiplier = 1.0 + 2.0 * distance_from_mid
            
            # Apply the adjusted noise
            adjusted_std = base_noise_std * variance_multiplier
            noise_factor = np.random.normal(self.config.NOISE_FACTOR_MEAN, adjusted_std)
            
            delivery_time *= noise_factor
            
        return delivery_time, is_outage 