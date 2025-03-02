"""Configuration management module."""
import yaml
from pathlib import Path


class Config:
    """Configuration manager for simulation parameters."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration with default or custom values."""
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Look for config file in project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            self.config_path = project_root / "config" / "simulation_config.yaml"
            
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found at {self.config_path}. "
                f"Make sure the file exists and you're running from the correct directory."
            )
            
        # Simulation parameters
        self.N_CENTERS = config["simulation"]["n_centers"]
        self.N_DAYS = config["simulation"]["n_days"]
        self.N_OUTAGES = config["simulation"]["n_outages"]
        
        # Delivery parameters
        self.BASE_TIME_SECONDS = config["delivery"]["base_time_seconds"]
        self.FIXED_TIME_PER_TOUR = config["delivery"]["fixed_time_per_tour"]
        self.TOURS_PER_CENTER_MEAN = config["delivery"]["tours_per_center_mean"]
        self.TOURS_PER_CENTER_STD = config["delivery"]["tours_per_center_std"]
        self.PARCELS_PER_TOUR_MEAN = config["delivery"]["parcels_per_tour_mean"]
        self.PARCELS_PER_TOUR_STD = config["delivery"]["parcels_per_tour_std"]
        self.MIN_PARCELS_PER_TOUR = config["delivery"]["min_parcels_per_tour"]
        self.MAX_PARCELS_PER_TOUR = config["delivery"]["max_parcels_per_tour"]
        
        # Routing parameters
        self.DYNAMIC_ROUTING_MIN_FACTOR = config["routing"]["dynamic_min_factor"]
        self.DYNAMIC_ROUTING_MAX_FACTOR = config["routing"]["dynamic_max_factor"]
        self.MANUAL_ROUTING_MIN_FACTOR = config["routing"]["manual_min_factor"]
        self.MANUAL_ROUTING_MAX_FACTOR = config["routing"]["manual_max_factor"]
        
        # Analysis parameters
        self.NOISE_FACTOR_MEAN = config["analysis"]["noise_factor_mean"]
        self.NOISE_FACTOR_STD = config["analysis"]["noise_factor_std"] 