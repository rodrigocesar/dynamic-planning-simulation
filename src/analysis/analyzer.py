"""Analysis module for routing outage simulation."""
import numpy as np
import pandas as pd


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