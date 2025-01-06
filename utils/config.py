import capymoa.drift.detectors as detectors

from capymoa.classifier import (OnlineBagging,
                                HoeffdingTree,
                                AdaptiveRandomForestClassifier,
                                NaiveBayes)
from utils.studd import STUDD

DETECTORS = {
    'ADWIN': detectors.ADWIN,
    'DDM': detectors.DDM,
    'CUSUM': detectors.CUSUM,
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
    'AdaptiveRandomForestClassifier': AdaptiveRandomForestClassifier,
    'NaiveBayes': NaiveBayes,
}

CLASSIFIER_PARAMS = {
    'OnlineBagging': {'ensemble_size': 100},
    'HoeffdingTree': {},
    'AdaptiveRandomForestClassifier': {'ensemble_size': 100, 'disable_drift_detection': True},
    'NaiveBayes': {},
}
