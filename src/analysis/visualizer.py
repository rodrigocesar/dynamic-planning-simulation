"""Visualization module for routing outage simulation."""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_results(df):
    """Create visualizations for the simulation results"""
    # Set style
    plt.style.use('default')
    sns.set_theme()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Distribution of delivery times by outage status
    plt.subplot(2, 2, 1)
    sns.boxplot(x='is_outage', y='delivery_time_per_parcel', data=df)
    plt.title('Delivery Time Distribution by Outage Status')
    plt.xlabel('Outage Status')
    plt.ylabel('Delivery Time per Parcel (seconds)')
    
    # 2. Scatter plot of parcel count vs delivery time
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='n_parcels', y='delivery_time_per_parcel', 
                    hue='is_outage', alpha=0.5)
    plt.title('Delivery Time vs Parcel Count')
    plt.xlabel('Number of Parcels')
    plt.ylabel('Delivery Time per Parcel (seconds)')
    
    # 3. Time series of average daily delivery times
    daily_avg = (df.groupby(['date', 'is_outage'])['delivery_time_per_parcel']
                .mean().reset_index())
    plt.subplot(2, 2, 3)
    sns.lineplot(data=daily_avg, x='date', y='delivery_time_per_parcel', 
                 hue='is_outage')
    plt.title('Average Daily Delivery Times')
    plt.xlabel('Date')
    plt.ylabel('Avg Delivery Time per Parcel (seconds)')
    plt.xticks(rotation=45)
    
    # 4. Tours per center analysis
    plt.subplot(2, 2, 4)
    tours_per_center = (df.groupby(['center_id', 'date'])
                       .size().reset_index(name='n_tours'))
    sns.boxplot(data=tours_per_center, y='n_tours')
    plt.title('Distribution of Daily Tours per Center')
    plt.ylabel('Number of Tours')
    
    plt.tight_layout()
    plt.show() 