import capymoa.drift.detectors as detectors

from capymoa.classifier import (OnlineBagging,
                                HoeffdingTree,
                                AdaptiveRandomForestClassifier,
                                NaiveBayes)
from utils.studd import STUDD

DETECTORS = {
    'ABCD': detectors.ABCD,
    'ADWIN': detectors.ADWIN,
    'CUSUM': detectors.CUSUM,
    'DDM': detectors.DDM,
    'EWMAChart': detectors.EWMAChart,
    'GeometricMovingAverage': detectors.GeometricMovingAverage,
    'HDDMAverage': detectors.HDDMAverage,
    'HDDMWeighted': detectors.HDDMWeighted,
    'PageHinkley': detectors.PageHinkley,
    'RDDM': detectors.RDDM,
    'SEED': detectors.SEED,
    'STEPD': detectors.STEPD,
    'STUDD': STUDD,
}

CLASSIFIERS = {
    'OnlineBagging': OnlineBagging,
    'HoeffdingTree': HoeffdingTree,
    'ARF': AdaptiveRandomForestClassifier,
    'NaiveBayes': NaiveBayes,
}

CLASSIFIER_PARAMS = {
    'OnlineBagging': {'ensemble_size': 25},
    'HoeffdingTree': {},
    'ARF': {'ensemble_size': 25, 'disable_drift_detection': True},
    'NaiveBayes': {},
}

DETECTOR_PARAM_SPACE = {
    'ABCD': {
        'delta_drift': [0.001, 0.002, 0.005],
        'model_id': ["pca", "kpca", "ae"],
    },
    'ADWIN': {'delta': [0.001, 0.002, 0.005]},
    'CUSUM': {'min_n_instances': [30, 50, 100, 300],
              'delta': [0.001, 0.002, 0.005],
              'lambda_': [50, 100, 150, 300]},
    'DDM': {
        'min_n_instances': [30, 50, 100, 300],
        'out_control_level': [2.5, 3.0, 2.25],
    },
    'EWMAChart': {
        'min_n_instances': [30, 50, 100, 300],
        'lambda_': [0.01, 0.001, 0.1, 0.2, 0.3],
    },

    'GeometricMovingAverage': {
        'min_n_instances': [30, 50, 100, 300],
        'lambda_': [1, 2, 3, 5],
        'alpha': [0.99, 0.995, 0.9]
    },

    'HDDMAverage': {
        'drift_confidence': [0.001, 0.002, 0.005],
        'test_type': ['Two-sided', 'One-sided']
    },

    'HDDMWeighted': {
        'drift_confidence': [0.001, 0.002, 0.005],
        'test_type': ['Two-sided', 'One-sided'],
        'lambda_': [0.05, 0.001, 0.1],
    },
    'PageHinkley': {
        'min_n_instances': [30, 50, 100, 300],
        'delta': [0.001, 0.002, 0.005, 0.01],
        'lambda_': [30, 50, 100, 300],
        'alpha': [0.99, 0.999, 0.995, 0.9]
    },

    'RDDM': {
        'min_n_instances': [30, 50, 100, 300],
        'drift_level': [1.9, 2, 2.1, 2.25, 2.5]
    },

    'SEED': {
        'delta': [0.001, 0.01, 0.05, 0.1],
        'epsilon_prime': [0.0025, 0.01, 0.005],
        'block_size': [32, 50, 100, 256],
        'alpha': [0.5, 0.6, 0.7, 0.8]
    },

    'STEPD': {
        'window_size': [30, 50, 100, 300],
        'alpha_drift': [0.001, 0.002, 0.003, 0.005],
    },

    'STUDD': {'min_n_instances': [250, 500, 1000, 2000]},
}
