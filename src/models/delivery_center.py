"""Module containing the DeliveryCenter class implementation."""
from src.utils.config import Config
import numpy as np
import random


class DeliveryCenter:
    """Represents a delivery center with its daily tours."""

    def __init__(self, center_id: int):
        """Initialize delivery center.
        
        Args:
            center_id: Unique identifier for the center
        """
        self.center_id = center_id
        self.config = Config()
        
        # Generate center-specific efficiency factors
        # Centers will have efficiency factors between 0.8 and 1.2
        # This creates a 40% variation in base delivery times between centers
        self.efficiency_factor = random.uniform(0.8, 1.2)
        
        # Geographical complexity factor (1-5 scale)
        # Higher values mean more complex delivery areas (rural, mountainous, etc.)
        self.geographical_complexity = random.uniform(1, 5)
        
    def generate_daily_tours(self) -> list[int]:
        """Generate number of parcels for each tour for one day.
        
        Returns:
            list[int]: Number of parcels for each tour
        """
        # Generate number of tours for this day
        n_tours = int(np.random.normal(
            self.config.TOURS_PER_CENTER_MEAN,
            self.config.TOURS_PER_CENTER_STD
        ))
        n_tours = max(1, n_tours)  # Ensure at least one tour
        
        # Generate parcels for each tour using log-normal distribution
        # This creates a right-skewed distribution with fewer extreme values
        tours = []
        for _ in range(n_tours):
            # Log-normal parameters to target mean around 200 with fewer extremes
            mu = 5.2  # Log-mean parameter
            sigma = 0.3  # Log-standard deviation parameter
            
            # Generate from log-normal distribution
            n_parcels = int(np.random.lognormal(mu, sigma))
            
            # Clip to valid range, but with tighter bounds to make extremes rarer
            n_parcels = max(
                self.config.MIN_PARCELS_PER_TOUR,
                min(self.config.MAX_PARCELS_PER_TOUR, n_parcels)
            )
            tours.append(n_parcels)
            
        return tours
        
    def calculate_base_delivery_time(self, n_parcels: int) -> float:
        """Calculate base delivery time for entire tour in seconds.
        
        Incorporates center-specific efficiency and geographical complexity.
        """
        # Base time calculation with center-specific factors
        # Less efficient centers (higher efficiency_factor) take longer
        # More complex geography (higher geographical_complexity) takes longer
        
        # Base time per parcel adjusted by center efficiency
        adjusted_time_per_parcel = self.config.BASE_TIME_SECONDS * self.efficiency_factor
        
        # Geography effect - more complex areas have higher fixed times
        geography_factor = 1 + (self.geographical_complexity - 1) / 10  # 1.0 to 1.4 range
        adjusted_fixed_time = self.config.FIXED_TIME_PER_TOUR * geography_factor
        
        # Calculate total delivery time
        return (adjusted_time_per_parcel * n_parcels + adjusted_fixed_time) 