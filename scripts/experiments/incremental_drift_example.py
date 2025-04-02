#!/usr/bin/env python3
"""
Example script demonstrating the use of the IncrementalDriftStream and IncrementalDriftSimulator.

This script shows how to:
1. Use the IncrementalDriftStream to apply incremental drift to an existing stream
2. Use the static simulate_incremental_drift method for DataFrame processing
3. Visualize the simulated drift
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from capymoa.evaluation.evaluation import ClassificationEvaluator
from capymoa.drift.eval_detector import EvaluateDriftDetector
from capymoa.stream.generator import AgrawalGenerator
from capymoa.stream import DriftStream

from utils.streams.incremental_drift import IncrementalDriftStream, IncrementalDriftSimulator
from utils.prequential_workflow import StreamingWorkflow
from utils.config import CLASSIFIERS, DETECTOR_SYNTH_PARAMS


def example_with_stream():
    """
    Example showing how to use IncrementalDriftStream with a base stream.
    """
    print("Running example with IncrementalDriftStream...")
    
    # Create a base stream
    base_stream = AgrawalGenerator(classification_function=1)
    
    # Create incremental drift stream
    drift_stream = IncrementalDriftStream(
        base_stream=base_stream,
        num_drifts=2,
        drift_lengths=1000,
        drop_drifting_feature=True
    )
    
    # Prepare the stream (collect initial data and fit the simulator)
    drift_stream.prepare(max_instances=5000)
    
    # Get schema and drift points
    schema = drift_stream.get_schema()
    drift_points = drift_stream.get_drifts()
    print(f"Drift points: {drift_points}")
    
    # Initialize a classifier and evaluator
    classifier = CLASSIFIERS['HoeffdingTree'](schema=schema)
    evaluator = ClassificationEvaluator(schema=schema, window_size=100)
    
    # Create a detector with optimized parameters
    detector_name = 'SEED'
    detector_params = DETECTOR_SYNTH_PARAMS['Agrawal'][detector_name]
    detector = DETECTOR_SYNTH_PARAMS['ALL'][detector_name]
    
    # Set up workflow
    workflow = StreamingWorkflow(
        model=classifier,
        evaluator=evaluator,
        detector=detector,
        use_window_perf=True
    )
    
    # Run prequential evaluation
    workflow.run_prequential(stream=drift_stream, max_size=10000)
    
    # Evaluate detector performance
    drift_eval = EvaluateDriftDetector(max_delay=1000)
    metrics = drift_eval.calc_performance(
        trues=drift_points,
        preds=workflow.drift_predictions,
        tot_n_instances=workflow.instances_processed
    )
    
    print(f"Detector performance: {metrics}")


def example_with_dataframe():
    """
    Example showing how to use the static method to simulate drift on a DataFrame.
    """
    print("\nRunning example with DataFrame simulation...")
    
    # Create a synthetic dataset
    np.random.seed(42)
    n_samples = 10000
    
    # Create features
    X = np.random.randn(n_samples, 5)
    
    # Create target with correlation to feature 2
    y = (X[:, 2] > 0).astype(int)
    
    # Add some noise
    y = np.logical_xor(y, np.random.random(n_samples) > 0.2).astype(int)
    
    # Create DataFrame with column names
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    print(f"Original dataset shape: {df.shape}")
    
    # Apply incremental drift simulation
    drifted_df, drifting_feature = IncrementalDriftSimulator.simulate_incremental_drift(
        concept=df,
        num_drifts=2,
        drift_lengths=1000,
        drifting_feature=None,  # Will be auto-selected based on correlation
        target_index=-1,
        method='pearson',
        drop_drifting_feature=False
    )
    
    print(f"Drifted dataset shape: {drifted_df.shape}")
    print(f"Selected drifting feature: {drifting_feature}")
    
    # Visualize the drift
    visualize_drift(df, drifted_df, drifting_feature)


def visualize_drift(original_df, drifted_df, drifting_feature):
    """
    Visualize the original and drifted data.
    
    Args:
        original_df: Original DataFrame
        drifted_df: DataFrame with simulated drift
        drifting_feature: Name of the drifting feature
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot original data
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(
        range(len(original_df)), 
        original_df[drifting_feature], 
        alpha=0.5, 
        s=3, 
        c=original_df['target']
    )
    ax1.set_title(f"Original {drifting_feature}")
    ax1.set_xlabel("Instance index")
    ax1.set_ylabel(drifting_feature)
    
    # Plot original target
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(
        range(len(original_df)), 
        original_df['target'], 
        alpha=0.5, 
        s=3, 
        c=original_df[drifting_feature]
    )
    ax2.set_title("Original target")
    ax2.set_xlabel("Instance index")
    ax2.set_ylabel("Target")
    
    # Plot drifted data
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(
        range(len(drifted_df)), 
        drifted_df[drifting_feature], 
        alpha=0.5, 
        s=3, 
        c=drifted_df['target']
    )
    ax3.set_title(f"Drifted {drifting_feature}")
    ax3.set_xlabel("Instance index")
    ax3.set_ylabel(drifting_feature)
    
    # Plot drifted target
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(
        range(len(drifted_df)), 
        drifted_df['target'], 
        alpha=0.5, 
        s=3, 
        c=drifted_df[drifting_feature]
    )
    ax4.set_title("Drifted target")
    ax4.set_xlabel("Instance index")
    ax4.set_ylabel("Target")
    
    plt.tight_layout()
    plt.savefig('assets/incremental_drift_visualization.png')
    print("Visualization saved to 'assets/incremental_drift_visualization.png'")


if __name__ == "__main__":
    # Create output directory if not exists
    import os
    os.makedirs('assets', exist_ok=True)
    
    # Run examples
    try:
        example_with_dataframe()
    except Exception as e:
        print(f"Error in DataFrame example: {e}")
    
    try:
        example_with_stream()
    except Exception as e:
        print(f"Error in stream example: {e}") 