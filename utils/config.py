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
    'ARF': {'ensemble_size': 10, 'disable_drift_detection': True},
    'NaiveBayes': {},
}
