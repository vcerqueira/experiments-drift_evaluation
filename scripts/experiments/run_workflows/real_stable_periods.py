from pprint import pprint

from capymoa.evaluation.evaluation import ClassificationEvaluator

from utils.prequential_workflow import StreamingWorkflow
from utils.streams.real import CAPYMOA_DATASETS
from utils.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_ENSEMBLE, DETECTOR_SYNTH_PARAMS

CLF = 'NaiveBayes'
# DATASET = 'Covtype'
DATASET = 'Bike'
# DATASET = 'Sensor'

stream = CAPYMOA_DATASETS[DATASET]()
sch = stream.get_schema()

stream = CAPYMOA_DATASETS[DATASET]()

evaluator = ClassificationEvaluator(schema=sch, window_size=1)
learner = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])
# student = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])

detector_ensemble = {k: DETECTORS[k](**DETECTOR_SYNTH_PARAMS['ALL'][k]) for k in DETECTOR_ENSEMBLE}

wf = StreamingWorkflow(model=learner,
                       evaluator=evaluator,
                       detector=detector_ensemble,
                       use_window_perf=False,
                       drift_simulator=None)

detectors_alarms = wf.run_prequential_ensemble_detectors(stream=stream)

MIN_GAP = 1000
MAX_DELAY = 250

stable_periods = wf.find_stable_periods(alarms_dict=detectors_alarms,
                                        min_gap=MIN_GAP,
                                        max_delay=MAX_DELAY,
                                        n=stream._length)

pprint(stable_periods)
