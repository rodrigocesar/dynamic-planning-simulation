"""Main entry point for the routing outage analysis simulation."""
from src.simulation.simulator import run_simulation
from src.analysis.analyzer import analyze_results
from src.analysis.visualizer import (
    plot_results,
    plot_additional_insights
)


def main():
    """Run the complete simulation and analysis pipeline."""
    # Run simulation
    results_df = run_simulation()
    
    # Analyze results
    outage_stats, effect_size = analyze_results(results_df)
    
    # Generate visualizations
    plot_results(results_df)
    plot_additional_insights(results_df)
    
    # Print results
    print("\nDelivery Time Statistics (seconds per parcel):")
    print(outage_stats)
    print(f"\nEffect Size (Cohen's d): {effect_size:.3f}")


if __name__ == "__main__":
    main() 