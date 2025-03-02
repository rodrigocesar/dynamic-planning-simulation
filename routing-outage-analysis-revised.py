import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Data Preparation with Outage Scenarios
def load_and_prepare_data(scenario='equal_severity'):
    """
    Load and prepare the dataset for analysis with specific outage scenarios.
    
    Parameters:
    scenario (str): Either 'equal_severity' or 'varying_severity'
    
    Returns:
    pandas.DataFrame: Prepared dataset
    """
    # Sample data generation to simulate the scenario
    np.random.seed(42)
    
    # Generate dates for 2024 (leap year)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31')
    
    # Define outage days (10 major incidents)
    outage_days = ['2024-02-15', '2024-03-22', '2024-04-10', '2024-05-18', 
                   '2024-06-07', '2024-07-25', '2024-08-12', '2024-09-30', 
                   '2024-10-21', '2024-11-15']
    
    # Define outage severity based on scenario
    if scenario == 'equal_severity':
        # Scenario 1: All outages have equal severity
        outage_severity = {day: 1 for day in outage_days}
    else:  # scenario == 'varying_severity'
        # Scenario 2: Outages have different severity levels
        # We'll use values 1-5 to represent different severity levels
        severities = [3, 5, 2, 4, 3, 5, 1, 4, 2, 3]
        outage_severity = {day: sev for day, sev in zip(outage_days, severities)}
    
    # Generate delivery center data
    center_ids = [f"DC_{i:03d}" for i in range(1, 31)]  # 30 delivery centers
    
    # Create empty lists to store data
    records = []
    
    # For each date and center, generate multiple tours
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        is_outage = date_str in outage_days
        severity = outage_severity.get(date_str, 0) if is_outage else 0
        
        # In both scenarios, all centers are affected on outage days
        # (total daily outages as specified)
        if is_outage:
            affected_centers = center_ids
        else:
            affected_centers = []
        
        for center_id in center_ids:
            # Center-specific characteristics (fixed effects)
            center_efficiency = np.random.normal(1, 0.15)  # Efficiency multiplier
            
            # Number of tours per day varies by center and day
            num_tours = np.random.randint(5, 20)
            
            for tour_id in range(1, num_tours + 1):
                # Parcels per tour: log-normal distribution to represent realistic parcel counts
                parcels = int(np.random.lognormal(4.5, 0.5))  # Centered around ~90 parcels
                
                # If it's a weekend, reduce parcel count
                if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    parcels = int(parcels * 0.7)
                
                # Baseline time per parcel (seconds)
                base_time_per_parcel = 60  # 1 minute per parcel baseline
                
                # Center effect
                center_effect = base_time_per_parcel * (1 / center_efficiency)
                
                # Volume effect (economies of scale)
                volume_effect = 120 / (1 + np.log(parcels))
                
                # Dynamic routing effect (reduces time per parcel)
                routing_effect = 0
                
                # If center is affected by outage, add routing effect
                # Severity now directly affects the magnitude of the effect
                if center_id in affected_centers:
                    if scenario == 'equal_severity':
                        # For equal severity, all outages have the same effect
                        routing_effect = base_time_per_parcel * 0.15  # 15% increase
                    else:
                        # For varying severity, the effect scales with severity
                        routing_effect = base_time_per_parcel * (0.05 * severity)  # 5-25% increase based on severity
                
                # Weather and traffic random variation
                random_variation = np.random.normal(0, 10)
                
                # Calculate final time per parcel
                time_per_parcel = center_effect + volume_effect + routing_effect + random_variation
                
                # Add heteroscedasticity - smaller parcel volumes have higher variance
                if parcels < 60:
                    time_per_parcel += np.random.normal(0, 20)
                
                # Generate random urban/rural classification
                urban_rural = np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.6, 0.3, 0.1])
                
                # Get courier experience level (years)
                courier_experience = np.random.randint(1, 16)  # 1-15 years
                
                # Calculate delivery distance (km)
                if urban_rural == 'Urban':
                    distance = np.random.uniform(5, 15)
                elif urban_rural == 'Suburban':
                    distance = np.random.uniform(10, 30)
                else:  # Rural
                    distance = np.random.uniform(20, 50)
                
                # Weather condition
                weather = np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], p=[0.7, 0.2, 0.05, 0.05])
                
                # Create record
                record = {
                    'date': date_str,
                    'center_id': center_id,
                    'tour_id': f"{center_id}_T{tour_id:03d}",
                    'parcels': parcels,
                    'time_per_parcel': max(10, time_per_parcel),  # Ensure no negative times
                    'is_outage': is_outage,
                    'outage_severity': severity,
                    'center_affected': center_id in affected_centers,
                    'weekday': date.weekday(),
                    'month': date.month,
                    'urban_rural': urban_rural,
                    'courier_experience': courier_experience,
                    'distance_km': distance,
                    'weather': weather,
                    'scenario': scenario
                }
                records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Add weekday name
    weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                     4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['weekday_name'] = df['weekday'].map(weekday_names)
    
    return df

# 2. Combine and Analyze Both Scenarios
def analyze_both_scenarios():
    """
    Analyze both outage severity scenarios and compare results.
    """
    # Load data for both scenarios
    df_equal = load_and_prepare_data(scenario='equal_severity')
    df_varying = load_and_prepare_data(scenario='varying_severity')
    
    # Combine datasets
    df_equal['scenario'] = 'Equal Severity'
    df_varying['scenario'] = 'Varying Severity'
    df_combined = pd.concat([df_equal, df_varying])
    
    # Simple visualization of outage effects
    plt.figure(figsize=(12, 8))
    
    # Overall outage effect by scenario
    plt.subplot(2, 2, 1)
    sns.boxplot(x='scenario', y='time_per_parcel', hue='is_outage', data=df_combined)
    plt.title('Outage Impact by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Time per Parcel (seconds)')
    plt.legend(title='Outage Day')
    
    # Equal severity analysis
    plt.subplot(2, 2, 3)
    sns.boxplot(x='is_outage', y='time_per_parcel', data=df_equal)
    plt.title('Equal Severity: Outage Impact')
    plt.xlabel('Outage Day')
    plt.ylabel('Time per Parcel (seconds)')
    
    # Varying severity analysis
    plt.subplot(2, 2, 4)
    # For varying severity, show by severity level
    severity_order = sorted(df_varying['outage_severity'].unique())
    sns.boxplot(x='outage_severity', y='time_per_parcel', data=df_varying)
    plt.title('Varying Severity: Impact by Severity Level')
    plt.xlabel('Outage Severity (0-5)')
    plt.ylabel('Time per Parcel (seconds)')
    
    plt.tight_layout()
    plt.savefig('scenario_comparison.png')
    
    return df_equal, df_varying, df_combined

# 3. Statistical Analysis for Each Scenario
def analyze_scenario(df, scenario_name):
    """
    Run statistical analysis for a specific scenario
    
    Parameters:
    df (pandas.DataFrame): Dataset for the scenario
    scenario_name (str): Name of the scenario for reporting
    """
    print(f"\n===== Statistical Analysis for {scenario_name} =====")
    
    # Simple t-test for difference in means
    outage_data = df[df['is_outage']]['time_per_parcel']
    no_outage_data = df[~df['is_outage']]['time_per_parcel']
    
    t_stat, p_val = stats.ttest_ind(outage_data, no_outage_data, equal_var=False)
    
    print("\nT-test for difference in mean time per parcel:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_val:.4f}")
    print(f"Mean with outage: {outage_data.mean():.2f} seconds")
    print(f"Mean without outage: {no_outage_data.mean():.2f} seconds")
    print(f"Difference: {outage_data.mean() - no_outage_data.mean():.2f} seconds")
    
    # Base model - simple outage effect
    base_formula = "time_per_parcel ~ is_outage"
    base_model = ols(base_formula, data=df).fit()
    
    print("\nBase Model (Outage Effect Only):")
    print(base_model.summary().tables[1])
    
    # Scenario-specific modeling
    if 'varying' in scenario_name.lower():
        # For varying severity, include severity as a factor
        severity_formula = "time_per_parcel ~ outage_severity"
        severity_model = ols(severity_formula, data=df).fit()
        
        print("\nSeverity Model:")
        print(severity_model.summary().tables[1])
        
        # Create severity categories for visualization
        df['severity_category'] = pd.Categorical(
            df['outage_severity'].apply(lambda x: f"Level {x}" if x > 0 else "No Outage"),
            categories=["No Outage"] + [f"Level {i}" for i in range(1, 6)],
            ordered=True
        )
        
        # Visualize severity effects
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='severity_category', y='time_per_parcel', data=df)
        plt.title(f'{scenario_name}: Time per Parcel by Outage Severity')
        plt.xlabel('Outage Severity')
        plt.ylabel('Time per Parcel (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{scenario_name.lower().replace(" ", "_")}_severity_effects.png')
    
    # Comprehensive model with controls (for both scenarios)
    control_formula = ("time_per_parcel ~ is_outage + "
                      "parcels + C(urban_rural) + C(weekday) + C(month) + "
                      "courier_experience + distance_km + C(weather) + C(center_id)")
    
    # For varying severity, replace is_outage with outage_severity
    if 'varying' in scenario_name.lower():
        control_formula = control_formula.replace("is_outage", "outage_severity")
    
    control_model = ols(control_formula, data=df).fit()
    
    print("\nFull Model (with controls):")
    print(control_model.summary().tables[1])
    
    # Analyze potential interaction effects
    interaction_models = {}
    
    # 1. Interaction with Urban/Rural
    urban_formula = "time_per_parcel ~ is_outage * C(urban_rural)"
    if 'varying' in scenario_name.lower():
        urban_formula = urban_formula.replace("is_outage", "C(severity_category)")
    
    urban_model = ols(urban_formula, data=df).fit()
    interaction_models['urban_rural'] = urban_model
    
    # 2. Interaction with parcel volume
    df['volume_category'] = pd.cut(df['parcels'], 
                                 bins=[0, 30, 60, 100, np.inf], 
                                 labels=['Very Low', 'Low', 'Medium', 'High'])
    
    volume_formula = "time_per_parcel ~ is_outage * C(volume_category)"
    if 'varying' in scenario_name.lower():
        volume_formula = volume_formula.replace("is_outage", "C(severity_category)")
    
    volume_model = ols(volume_formula, data=df).fit()
    interaction_models['volume'] = volume_model
    
    # 3. Interaction with day of week
    weekday_formula = "time_per_parcel ~ is_outage * C(weekday_name)"
    if 'varying' in scenario_name.lower():
        weekday_formula = weekday_formula.replace("is_outage", "C(severity_category)")
    
    weekday_model = ols(weekday_formula, data=df).fit()
    interaction_models['weekday'] = weekday_model
    
    # Visualize key interactions
    plt.figure(figsize=(15, 10))
    
    # Urban/Rural interaction
    plt.subplot(2, 2, 1)
    if 'varying' in scenario_name.lower():
        urban_data = df.groupby(['urban_rural', 'severity_category'])['time_per_parcel'].mean().reset_index()
        sns.barplot(x='urban_rural', y='time_per_parcel', hue='severity_category', data=urban_data)
    else:
        urban_data = df.groupby(['urban_rural', 'is_outage'])['time_per_parcel'].mean().reset_index()
        sns.barplot(x='urban_rural', y='time_per_parcel', hue='is_outage', data=urban_data)
    plt.title(f'{scenario_name}: Impact by Urban/Rural Classification')
    plt.xlabel('Area Type')
    plt.ylabel('Avg Time per Parcel (seconds)')
    plt.legend(title='Outage Status', loc='upper right')
    
    # Volume interaction
    plt.subplot(2, 2, 2)
    if 'varying' in scenario_name.lower():
        volume_data = df.groupby(['volume_category', 'severity_category'])['time_per_parcel'].mean().reset_index()
        # Use only No Outage and select severity levels for clarity
        severity_to_plot = ["No Outage", "Level 1", "Level 3", "Level 5"]
        volume_data = volume_data[volume_data['severity_category'].isin(severity_to_plot)]
        sns.barplot(x='volume_category', y='time_per_parcel', hue='severity_category', data=volume_data)
    else:
        volume_data = df.groupby(['volume_category', 'is_outage'])['time_per_parcel'].mean().reset_index()
        sns.barplot(x='volume_category', y='time_per_parcel', hue='is_outage', data=volume_data)
    plt.title(f'{scenario_name}: Impact by Parcel Volume')
    plt.xlabel('Parcel Volume Category')
    plt.ylabel('Avg Time per Parcel (seconds)')
    plt.legend(title='Outage Status', loc='upper right')
    
    # Weekday interaction
    plt.subplot(2, 2, 3)
    if 'varying' in scenario_name.lower():
        weekday_data = df.groupby(['weekday_name', 'is_outage'])['time_per_parcel'].mean().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_data['weekday_name'] = pd.Categorical(weekday_data['weekday_name'], 
                                                    categories=day_order, ordered=True)
        weekday_data = weekday_data.sort_values('weekday_name')
        sns.barplot(x='weekday_name', y='time_per_parcel', hue='is_outage', data=weekday_data)
    else:
        weekday_data = df.groupby(['weekday_name', 'is_outage'])['time_per_parcel'].mean().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_data['weekday_name'] = pd.Categorical(weekday_data['weekday_name'], 
                                                    categories=day_order, ordered=True)
        weekday_data = weekday_data.sort_values('weekday_name')
        sns.barplot(x='weekday_name', y='time_per_parcel', hue='is_outage', data=weekday_data)
    plt.title(f'{scenario_name}: Impact by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Avg Time per Parcel (seconds)')
    plt.legend(title='Outage Status', loc='upper right')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{scenario_name.lower().replace(" ", "_")}_interactions.png')
    
    return {
        'base_model': base_model,
        'control_model': control_model,
        'interaction_models': interaction_models,
        'severity_model': severity_model if 'varying' in scenario_name.lower() else None
    }

# 4. Compare and Summarize Both Scenarios
def compare_scenarios(equal_models, varying_models, df_equal, df_varying):
    """
    Compare the results of both scenarios and provide insights.
    """
    print("\n===== Scenario Comparison =====")
    
    # Basic impact comparison
    equal_impact = equal_models['base_model'].params['is_outage[T.True]']
    equal_impact_pct = (equal_impact / df_equal[~df_equal['is_outage']]['time_per_parcel'].mean()) * 100
    
    varying_base_impact = varying_models['base_model'].params['is_outage[T.True]']
    varying_base_impact_pct = (varying_base_impact / df_varying[~df_varying['is_outage']]['time_per_parcel'].mean()) * 100
    
    print("\n1. Basic Outage Impact Comparison:")
    print(f"   - Equal Severity: {equal_impact:.2f} seconds ({equal_impact_pct:.2f}% increase)")
    print(f"   - Varying Severity (Base Effect): {varying_base_impact:.2f} seconds ({varying_base_impact_pct:.2f}% increase)")
    
    # For varying severity, analyze the incremental effect by severity level
    if varying_models['severity_model'] is not None:
        print("\n2. Varying Severity Incremental Effects:")
        for level in range(1, 6):
            param_name = f'outage_severity[{level}.0]'
            if param_name in varying_models['severity_model'].params:
                effect = varying_models['severity_model'].params[param_name]
                effect_pct = (effect / df_varying[df_varying['outage_severity'] == 0]['time_per_parcel'].mean()) * 100
                print(f"   - Severity Level {level}: {effect:.2f} seconds ({effect_pct:.2f}% increase)")
    
    # Compare controlled models
    equal_controlled = equal_models['control_model'].params['is_outage[T.True]']
    equal_controlled_pct = (equal_controlled / df_equal[~df_equal['is_outage']]['time_per_parcel'].mean()) * 100
    
    print("\n3. Controlled Model Comparison:")
    print(f"   - Equal Severity (with controls): {equal_controlled:.2f} seconds ({equal_controlled_pct:.2f}% increase)")
    
    if 'outage_severity' in varying_models['control_model'].params:
        print("   - Varying Severity (with controls): See incremental effects by severity level")
        for level in range(1, 6):
            param_name = f'outage_severity[{level}.0]'
            if param_name in varying_models['control_model'].params:
                effect = varying_models['control_model'].params[param_name]
                effect_pct = (effect / df_varying[df_varying['outage_severity'] == 0]['time_per_parcel'].mean()) * 100
                print(f"     * Severity Level {level}: {effect:.2f} seconds ({effect_pct:.2f}% increase)")
    
    # Analyze interaction effects
    print("\n4. Key Interaction Findings:")
    
    # Urban/Rural interaction
    urban_rural_equal = equal_models['interaction_models']['urban_rural']
    urban_rural_varying = varying_models['interaction_models']['urban_rural']
    
    print("\n   a. Urban/Rural Interaction:")
    print("      - Equal Severity: Significant interaction? ", 
          "Yes" if any("is_outage" in param and "urban_rural" in param and p < 0.05 
                     for param, p in zip(urban_rural_equal.pvalues.index, urban_rural_equal.pvalues)) 
          else "No")
    
    print("      - Varying Severity: Significant interaction? ", 
          "Yes" if any("severity_category" in param and "urban_rural" in param and p < 0.05 
                     for param, p in zip(urban_rural_varying.pvalues.index, urban_rural_varying.pvalues)) 
          else "No")
    
    # Volume interaction
    volume_equal = equal_models['interaction_models']['volume']
    volume_varying = varying_models['interaction_models']['volume']
    
    print("\n   b. Parcel Volume Interaction:")
    print("      - Equal Severity: Significant interaction? ", 
          "Yes" if any("is_outage" in param and "volume_category" in param and p < 0.05 
                     for param, p in zip(volume_equal.pvalues.index, volume_equal.pvalues)) 
          else "No")
    
    print("      - Varying Severity: Significant interaction? ", 
          "Yes" if any("severity_category" in param and "volume_category" in param and p < 0.05 
                     for param, p in zip(volume_varying.pvalues.index, volume_varying.pvalues)) 
          else "No")
    
    # Create comparison visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Overall effect comparison
    plt.subplot(2, 2, 1)
    scenario_data = pd.DataFrame({
        'Scenario': ['Equal Severity', 'Varying Severity (Average)'],
        'Effect': [equal_impact, varying_base_impact],
        'Effect_Pct': [equal_impact_pct, varying_base_impact_pct]
    })
    sns.barplot(x='Scenario', y='Effect', data=scenario_data)
    plt.title('Overall Outage Effect by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Effect on Time per Parcel (seconds)')
    
    # 2. Severity breakdown for varying scenario
    plt.subplot(2, 2, 2)
    if varying_models['severity_model'] is not None:
        severity_effects = []
        for level in range(0, 6):
            if level == 0:
                effect = 0  # Baseline (no outage)
            else:
                param_name = f'outage_severity[{level}.0]'
                effect = varying_models['severity_model'].params.get(param_name, 0)
            severity_effects.append({'Severity': f'Level {level}', 'Effect': effect})
        
        severity_df = pd.DataFrame(severity_effects)
        sns.barplot(x='Severity', y='Effect', data=severity_df)
        plt.title('Effect by Severity Level (Varying Scenario)')
        plt.xlabel('Outage Severity')
        plt.ylabel('Effect on Time per Parcel (seconds)')
    
    # 3. Compare urban vs rural effects
    plt.subplot(2, 2, 3)
    # Extract urban/rural effects
    urban_effects = []
    
    # Equal severity
    urban_equal = df_equal[df_equal['urban_rural'] == 'Urban']
    suburban_equal = df_equal[df_equal['urban_rural'] == 'Suburban']
    rural_equal = df_equal[df_equal['urban_rural'] == 'Rural']
    
    urban_effect_equal = urban_equal[urban_equal['is_outage']]['time_per_parcel'].mean() - \
                        urban_equal[~urban_equal['is_outage']]['time_per_parcel'].mean()
    
    suburban_effect_equal = suburban_equal[suburban_equal['is_outage']]['time_per_parcel'].mean() - \
                           suburban_equal[~suburban_equal['is_outage']]['time_per_parcel'].mean()
    
    rural_effect_equal = rural_equal[rural_equal['is_outage']]['time_per_parcel'].mean() - \
                        rural_equal[~rural_equal['is_outage']]['time_per_parcel'].mean()
    
    # Varying severity (using average effect)
    urban_varying = df_varying[df_varying['urban_rural'] == 'Urban']
    suburban_varying = df_varying[df_varying['urban_rural'] == 'Suburban']
    rural_varying = df_varying[df_varying['urban_rural'] == 'Rural']
    
    urban_effect_varying = urban_varying[urban_varying['is_outage']]['time_per_parcel'].mean() - \
                          urban_varying[~urban_varying['is_outage']]['time_per_parcel'].mean()
    
    suburban_effect_varying = suburban_varying[suburban_varying['is_outage']]['time_per_parcel'].mean() - \
                             suburban_varying[~suburban_varying['is_outage']]['time_per_parcel'].mean()
    
    rural_effect_varying = rural_varying[rural_varying['is_outage']]['time_per_parcel'].mean() - \
                          rural_varying[~rural_varying['is_outage']]['time_per_parcel'].mean()
    
    urban_effects = pd.DataFrame({
        'Area': ['Urban', 'Urban', 'Suburban', 'Suburban', 'Rural', 'Rural'],
        'Scenario': ['Equal', 'Varying', 'Equal', 'Varying', 'Equal', 'Varying'],
        'Effect': [urban_effect_equal, urban_effect_varying, 
                 suburban_effect_equal, suburban_effect_varying,
                 rural_effect_equal, rural_effect_varying]
    })
    
    sns.barplot(x='Area', y='Effect', hue='Scenario', data=urban_effects)
    plt.title('Urban/Rural Effect Comparison')
    plt.xlabel('Area Type')
    plt.ylabel('Effect on Time per Parcel (seconds)')
    
    # 4. Compare volume effects
    plt.subplot(2, 2, 4)
    # Extract volume effects
    volume_effects = []
    
    for vol_cat in ['Very Low', 'Low', 'Medium', 'High']:
        # Equal severity
        vol_equal = df_equal[df_equal['volume_category'] == vol_cat]
        vol_effect_equal = vol_equal[vol_equal['is_outage']]['time_per_parcel'].mean() - \
                          vol_equal[~vol_equal['is_outage']]['time_per_parcel'].mean()
        
        # Varying severity
        vol_varying = df_varying[df_varying['volume_category'] == vol_cat]
        vol_effect_varying = vol_varying[vol_varying['is_outage']]['time_per_parcel'].mean() - \
                            vol_varying[~vol_varying['is_outage']]['time_per_parcel'].mean()
        
        volume_effects.append({'Volume': vol_cat, 'Scenario': 'Equal', 'Effect': vol_effect_equal})
        volume_effects.append({'Volume': vol_cat, 'Scenario': 'Varying', 'Effect': vol_effect_varying})
    
    volume_effects_df = pd.DataFrame(volume_effects)
    sns.barplot(x='Volume', y='Effect', hue='Scenario', data=volume_effects_df)
    plt.title('Volume Category Effect Comparison')
    plt.xlabel('Parcel Volume Category')
    plt.ylabel('Effect on Time per Parcel (seconds)')
    
    plt.tight_layout()
    plt.savefig('scenario_effect_comparison.png')
    
    # Determine which scenario is better for modeling
    print("\n5. Model Evaluation:")
    
    equal_r2 = equal_models['control_model'].rsquared
    varying_r2 = varying_models['control_model'].rsquared
    
    equal_aic = equal_models['control_model'].aic
    varying_aic = varying_models['control_model'].aic
    
    print(f"   - Equal Severity Model: R-squared = {equal_r2:.4f}, AIC = {equal_aic:.2f}")
    print(f"   - Varying Severity Model: R-squared = {varying_r2:.4f}, AIC = {varying_aic:.2f}")
    print(f"   - Better model based on AIC: {'Varying Severity' if varying_aic < equal_aic else 'Equal Severity'}")
    
    # Final summary and recommendations
    print("\n===== Analysis Recommendations =====")
    
    print("""
Based on the analysis of both scenarios, here are key recommendations:

1. Choice of Modeling Approach:
   - If outages are believed to have equal impact, use the simpler equal