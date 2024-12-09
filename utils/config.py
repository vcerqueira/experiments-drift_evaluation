import capymoa.drift.detectors as detectors

from capymoa.classifier import OnlineBagging, HoeffdingTree

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
}

CLASSIFIERS = {
    'OnlineBagging': OnlineBagging,
    'HoeffdingTree': HoeffdingTree,
}
