import seaborn as sns
import matplotlib.pyplot as plt

def plot_results(df):
    """Create visualizations for the simulation results"""
    # Set style
    plt.style.use('seaborn')
    
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
    daily_avg = df.groupby(['date', 'is_outage'])['delivery_time_per_parcel'].mean().reset_index()
    plt.subplot(2, 2, 3)
    sns.lineplot(data=daily_avg, x='date', y='delivery_time_per_parcel', 
                 hue='is_outage')
    plt.title('Average Daily Delivery Times')
    plt.xlabel('Date')
    plt.ylabel('Avg Delivery Time per Parcel (seconds)')
    plt.xticks(rotation=45)
    
    # 4. Center efficiency analysis
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='efficiency_factor', y='delivery_time_per_parcel',
                    hue='is_outage', size='geographical_complexity', alpha=0.5)
    plt.title('Delivery Time vs Center Efficiency')
    plt.xlabel('Center Efficiency Factor')
    plt.ylabel('Delivery Time per Parcel (seconds)')
    
    plt.tight_layout()
    plt.show()

def plot_additional_insights(df):
    """Create additional visualizations for deeper analysis"""
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Violin plot of delivery times by outage status
    plt.subplot(1, 3, 1)
    sns.violinplot(x='is_outage', y='delivery_time_per_parcel', data=df)
    plt.title('Delivery Time Distribution (Violin Plot)')
    
    # 2. Center performance comparison
    center_stats = df.groupby(['center_id', 'is_outage'])['delivery_time_per_parcel'].mean().unstack()
    plt.subplot(1, 3, 2)
    plt.scatter(center_stats[False], center_stats[True])
    plt.plot([center_stats.min().min(), center_stats.max().max()], 
             [center_stats.min().min(), center_stats.max().max()], 'r--')
    plt.title('Center Performance:\nNormal vs Outage Days')
    plt.xlabel('Normal Days (avg seconds/parcel)')
    plt.ylabel('Outage Days (avg seconds/parcel)')
    
    # 3. Geographical complexity impact
    plt.subplot(1, 3, 3)
    sns.boxplot(x='geographical_complexity', y='delivery_time_per_parcel', 
                hue='is_outage', data=df)
    plt.title('Impact of Geographical Complexity')
    plt.xlabel('Geographical Complexity (binned)')
    plt.ylabel('Delivery Time per Parcel (seconds)')
    
    plt.tight_layout()
    plt.show() 