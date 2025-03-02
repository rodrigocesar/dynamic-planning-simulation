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
            improvement_factor = random.uniform(
                self.config.DYNAMIC_ROUTING_MIN_FACTOR,
                self.config.DYNAMIC_ROUTING_MAX_FACTOR
            )
            delivery_time = base_time * improvement_factor
        else:
            manual_factor = random.uniform(
                self.config.MANUAL_ROUTING_MIN_FACTOR,
                self.config.MANUAL_ROUTING_MAX_FACTOR
            )
            delivery_time = base_time * manual_factor
            
        if add_noise:
            noise_factor = np.random.normal(
                self.config.NOISE_FACTOR_MEAN,
                self.config.NOISE_FACTOR_STD
            )
            delivery_time *= noise_factor
            
        return delivery_time, is_outage 