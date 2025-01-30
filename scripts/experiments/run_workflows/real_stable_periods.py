from pprint import pprint

from capymoa.evaluation.evaluation import ClassificationEvaluator

from utils.prequential_workflow import StreamingWorkflow
from utils.streams.real import CAPYMOA_DATASETS
from utils.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_ENSEMBLE, DETECTOR_SYNTH_PARAMS

CLF = 'ARF'
# DATASET = 'Electricity'
# DATASET = 'Covtype'
DATASET = 'Sensor'

stream = CAPYMOA_DATASETS[DATASET]()
sch = stream.get_schema()

stream = CAPYMOA_DATASETS[DATASET]()

evaluator = ClassificationEvaluator(schema=sch, window_size=1)
learner = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])

detector_ensemble = {k: DETECTORS[k](**DETECTOR_SYNTH_PARAMS['ALL'][k]) for k in DETECTOR_ENSEMBLE}

wf = StreamingWorkflow(model=learner,
                       evaluator=evaluator,
                       detector=detector_ensemble,
                       use_window_perf=False,
                       drift_simulator=None)

detectors_alarms = wf.run_prequential_ensemble_detectors(stream=stream)
n_alarms = {k: len(detectors_alarms[k]) for k in detectors_alarms}
pprint(n_alarms)

MIN_GAP = 10000
MAX_DELAY = 500

stable_periods = wf.find_stable_periods(alarms_dict=detectors_alarms,
                                        min_gap=MIN_GAP,
                                        max_delay=MAX_DELAY,
                                        n=stream._length)

pprint(stable_periods)



{
    'Covtype': [(224000, 233806), (517400, 526300)],
    'Sensor': [(7304, 19200),
               (170504, 193300),
               (513480, 523900),
               (536840, 548700),
               (593512, 609000),
               (719000, 729700),
               (808939, 826000),
               (861300, 874600),
               (889800, 901500),
               (955752, 966000),
               (1068000, 1082800),
               (1132800, 1142900),
               (1257000, 1269400),
               (1291000, 1305600),
               (1376300, 1389200),
               (1467000, 1478600),
               (1631242, 1642100),
               (1679048, 1690900),
               (1865640, 1879500),
               (2098700, 2110900)]
}
