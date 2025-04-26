"""
Script for analyzing hyperparameter tuning results for drift detectors.
Uses leave-one-out cross-validation to find optimal configurations.
"""

from ast import literal_eval
from pathlib import Path
from typing import Dict, List
import pprint

import pandas as pd

DRIFT_TYPE = 'GRADUAL'

file_path = Path(__file__).parent.parent.parent.parent / 'assets' / 'results' / f'hypertuning,{DRIFT_TYPE}.csv'


def get_best_configs(
        perf_data: pd.DataFrame,
        stream_list: List[str],
        detector_list: List[str]
) -> Dict[str, Dict[str, Dict]]:
    """
    Find the best configurations for each detector using leave-one-out cross-validation.
    
    Args:
        perf_data: DataFrame with performance data
        stream_list: List of stream names
        detector_list: List of detector names
        
    Returns:
        Dictionary mapping stream names to detector configurations
    """
    loo_best_configs = {}

    for stream in stream_list + ['ALL']:
        print(f"Processing leave-one-out for stream: {stream}")

        if stream == 'ALL':
            perf_loo = perf_data
        else:
            perf_loo = perf_data.query(f'stream!="{stream}"')

        detectors_f1 = {}
        for detector in detector_list:
            print(f"Finding best config for detector: {detector}")

            perf_dct = perf_loo.query(f'detector=="{detector}"')

            # group by parameters and rank by f1
            params_f1 = perf_dct.groupby('params').mean(numeric_only=True)['f1']

            if not params_f1.empty:
                best_config = literal_eval(params_f1.sort_values().index[-1])
                detectors_f1[detector] = best_config
            else:
                print(f"No data found for detector {detector}")
                detectors_f1[detector] = {}

        loo_best_configs[stream] = detectors_f1

    return loo_best_configs


def main() -> None:
    """Main function to run the hyperparameter analysis."""
    perf = pd.read_csv(file_path)

    stream_list = perf['stream'].unique().tolist()
    detector_list = perf['detector'].unique().tolist()

    loo_best_configs = get_best_configs(perf, stream_list, detector_list)

    print("Best configurations found:")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(loo_best_configs)


if __name__ == "__main__":
    main()
