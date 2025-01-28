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

# AFTERWARDS, PICK THE TOP 3 BASED ON SYNTHETIC EXPERIMENTS
# DETECTOR_ENSEMBLE = ['SEED', 'ADWIN', 'HDDMAverage', 'ABCD']
DETECTOR_ENSEMBLE = ['SEED', 'ADWIN', 'HDDMAverage']

CLASSIFIERS = {
    # 'OnlineBagging': OnlineBagging,
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
        'delta_drift': [0.001, 0.002, 0.005, 0.01],
        'model_id': ["pca", "kpca", "ae"],
    },
    'ADWIN': {'delta': [0.001, 0.002, 0.005, 0.01, 0.0005, 0.0001]},
    'CUSUM': {'min_n_instances': [30, 50, 100, 300, 500, 1000, 2000],
              'delta': [0.001, 0.002, 0.005, 0.0001, 0.01],
              'lambda_': [50, 100, 150, 300, 500, 1000, 2000]},
    'DDM': {
        'min_n_instances': [30, 50, 100, 300, 500, 1000, 2000],
        'out_control_level': [1.75, 2, 2.5, 3.0, 2.25, 2.75, 3.5, 4],
    },
    'EWMAChart': {
        'min_n_instances': [50, 100, 300, 500, 1000, 2000, 3000, 5000, 10000],
        'lambda_': [0.9, 0.01, 0.001, 0.1, 0.005, 0.002, 0.0001],
    },

    'GeometricMovingAverage': {
        'min_n_instances': [30, 50, 100, 300, 500, 1000, 2000],
        'lambda_': [0.001, 0.002, 0.01, 0.1, 0.5, 1, 2, 3, 5],
        'alpha': [0.99, 0.995, 0.9, 0.8, 0.7, 0.5, 0.1, 0.01]
    },

    'HDDMAverage': {
        'drift_confidence': [0.001, 0.002, 0.005, 0.01, 0.0001],
        'test_type': ['Two-sided', 'One-sided']
    },

    'HDDMWeighted': {
        'drift_confidence': [0.001, 0.002, 0.005, 0.01, 0.0001],
        'test_type': ['Two-sided', 'One-sided'],
        'lambda_': [0.05, 0.001, 0.1, 0.01, 0.0001],
    },
    'PageHinkley': {
        'min_n_instances': [30, 50, 100, 300, 500, 1000, 2000],
        'delta': [0.001, 0.002, 0.005, 0.01, 0.0001, 0.1],
        'lambda_': [30, 50, 100, 300, 500, 1000, 2000],
        'alpha': [0.99, 0.999, 0.995, 0.9, 0.8, 0.5]
    },

    'RDDM': {
        'min_n_instances': [30, 50, 100, 300, 500, 1000, 2000],
        'drift_level': [1.9, 2, 2.1, 2.25, 2.5, 3, 1.5, 1.75]
    },
    'SEED': {
        'delta': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'epsilon_prime': [0.0025, 0.01, 0.005, 0.0075],
        'block_size': [32, 50, 100, 150, 256],
        'alpha': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    },

    'STEPD': {
        'window_size': [30, 50, 100, 300, 500, 1000],
        'alpha_drift': [0.001, 0.002, 0.003, 0.005, 0.01, 0.1],
    },

    'STUDD': {'min_n_instances': [250, 500, 1000, 2000, 3000, 5000, 10000]},
}

# OPTIMIZED USING LOO
DETECTOR_SYNTH_PARAMS = {'ALL': {'ABCD': {'delta_drift': 0.002, 'model_id': 'pca'},
                                 'ADWIN': {'delta': 0.002},
                                 'CUSUM': {'delta': 0.0001, 'lambda_': 50, 'min_n_instances': 500},
                                 'DDM': {'min_n_instances': 50, 'out_control_level': 1.75},
                                 'EWMAChart': {'lambda_': 0.005, 'min_n_instances': 5000},
                                 'GeometricMovingAverage': {'alpha': 0.7,
                                                            'lambda_': 0.01,
                                                            'min_n_instances': 2000},
                                 'HDDMAverage': {'drift_confidence': 0.01, 'test_type': 'Two-sided'},
                                 'HDDMWeighted': {'drift_confidence': 0.01,
                                                  'lambda_': 0.05,
                                                  'test_type': 'One-sided'},
                                 'PageHinkley': {'alpha': 0.999,
                                                 'delta': 0.001,
                                                 'lambda_': 50,
                                                 'min_n_instances': 1000},
                                 'RDDM': {'drift_level': 1.5, 'min_n_instances': 500},
                                 'SEED': {'alpha': 0.8,
                                          'block_size': 100,
                                          'delta': 0.05,
                                          'epsilon_prime': 0.0075},
                                 'STEPD': {'alpha_drift': 0.002, 'window_size': 100},
                                 'STUDD': {'min_n_instances': 2000}},
                         'Agrawal': {'ABCD': {'delta_drift': 0.002, 'model_id': 'pca'},
                                     'ADWIN': {'delta': 0.0005},
                                     'CUSUM': {'delta': 0.001, 'lambda_': 150, 'min_n_instances': 1000},
                                     'DDM': {'min_n_instances': 50, 'out_control_level': 1.75},
                                     'EWMAChart': {'lambda_': 0.005, 'min_n_instances': 5000},
                                     'GeometricMovingAverage': {'alpha': 0.7,
                                                                'lambda_': 0.01,
                                                                'min_n_instances': 2000},
                                     'HDDMAverage': {'drift_confidence': 0.01,
                                                     'test_type': 'Two-sided'},
                                     'HDDMWeighted': {'drift_confidence': 0.01,
                                                      'lambda_': 0.05,
                                                      'test_type': 'One-sided'},
                                     'PageHinkley': {'alpha': 0.999,
                                                     'delta': 0.001,
                                                     'lambda_': 50,
                                                     'min_n_instances': 1000},
                                     'RDDM': {'drift_level': 1.5, 'min_n_instances': 500},
                                     'SEED': {'alpha': 0.2,
                                              'block_size': 100,
                                              'delta': 0.1,
                                              'epsilon_prime': 0.01},
                                     'STEPD': {'alpha_drift': 0.002, 'window_size': 30},
                                     'STUDD': {'min_n_instances': 1000}},
                         'SEA': {'ABCD': {'delta_drift': 0.002, 'model_id': 'pca'},
                                 'ADWIN': {'delta': 0.002},
                                 'CUSUM': {'delta': 0.0001, 'lambda_': 100, 'min_n_instances': 1000},
                                 'DDM': {'min_n_instances': 50, 'out_control_level': 2},
                                 'EWMAChart': {'lambda_': 0.005, 'min_n_instances': 5000},
                                 'GeometricMovingAverage': {'alpha': 0.01,
                                                            'lambda_': 0.001,
                                                            'min_n_instances': 2000},
                                 'HDDMAverage': {'drift_confidence': 0.01, 'test_type': 'Two-sided'},
                                 'HDDMWeighted': {'drift_confidence': 0.005,
                                                  'lambda_': 0.1,
                                                  'test_type': 'Two-sided'},
                                 'PageHinkley': {'alpha': 0.999,
                                                 'delta': 0.001,
                                                 'lambda_': 50,
                                                 'min_n_instances': 1000},
                                 'RDDM': {'drift_level': 2.25, 'min_n_instances': 50},
                                 'SEED': {'alpha': 0.3,
                                          'block_size': 150,
                                          'delta': 0.01,
                                          'epsilon_prime': 0.01},
                                 'STEPD': {'alpha_drift': 0.001, 'window_size': 30},
                                 'STUDD': {'min_n_instances': 2000}},
                         'STAGGER': {'ABCD': {'delta_drift': 0.01, 'model_id': 'pca'},
                                     'ADWIN': {'delta': 0.002},
                                     'CUSUM': {'delta': 0.0001, 'lambda_': 50, 'min_n_instances': 500},
                                     'DDM': {'min_n_instances': 50, 'out_control_level': 1.75},
                                     'EWMAChart': {'lambda_': 0.005, 'min_n_instances': 5000},
                                     'GeometricMovingAverage': {'alpha': 0.9,
                                                                'lambda_': 0.1,
                                                                'min_n_instances': 50},
                                     'HDDMAverage': {'drift_confidence': 0.01,
                                                     'test_type': 'Two-sided'},
                                     'HDDMWeighted': {'drift_confidence': 0.01,
                                                      'lambda_': 0.05,
                                                      'test_type': 'One-sided'},
                                     'PageHinkley': {'alpha': 0.999,
                                                     'delta': 0.001,
                                                     'lambda_': 50,
                                                     'min_n_instances': 1000},
                                     'RDDM': {'drift_level': 1.5, 'min_n_instances': 500},
                                     'SEED': {'alpha': 0.8,
                                              'block_size': 100,
                                              'delta': 0.05,
                                              'epsilon_prime': 0.0075},
                                     'STEPD': {'alpha_drift': 0.002, 'window_size': 100},
                                     'STUDD': {'min_n_instances': 2000}}}
