import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator

from utils.streams.inject_drift import DriftSimulator
from utils.evaluate import EvaluateDetector
from utils.prequential_workflow import StreamingWorkflow
from utils.streams.real import CAPYMOA_DATASETS
from utils.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_ENSEMBLE

CLF = 'ARF'
DATASET = 'Electricity'
USE_WINDOW = False
MAX_DELAY_PERC = 0.1
N_DRIFTS = 50
DRIFT_ON_X = False
DRIFT_ON_Y = True
WINDOW_MODE = 'WINDOW' if USE_WINDOW else 'POINT'
DRIFT_TYPE = 'ABRUPT@X' if DRIFT_ON_X else 'ABRUPT@Y'

stream = CAPYMOA_DATASETS[DATASET]()
sch = stream.get_schema()
n = stream._length

max_delay = int(n * MAX_DELAY_PERC)

stream = CAPYMOA_DATASETS[DATASET]()

evaluator = ClassificationEvaluator(schema=sch, window_size=1)
learner = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])
student = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])

# todo add best params
detector_ensemble = {k: DETECTORS[k]() for k in DETECTOR_ENSEMBLE}

wf = StreamingWorkflow(model=learner,
                       evaluator=evaluator,
                       detector=detector_ensemble,
                       use_window_perf=USE_WINDOW,
                       drift_simulator=None)

wf.run_prequential_ensemble_detectors(stream=stream)

perf = pd.DataFrame(detector_perf).T
perf.to_csv(f'assets/results/{DATASET},{DRIFT_TYPE},{CLF},{WINDOW_MODE}.csv')
