import numpy as np
import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator

from utils.streams.synth import CustomDriftStream
from utils.evaluate import EvaluateDetector
from utils.prequential_workflow import StreamingWorkflow
from utils.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_SYNTH_PARAMS

CLF = 'ARF'
GENERATOR = 'SEA'
USE_WINDOW = False
MAX_DELAY = 1000
N_DRIFTS = 30
DRIFT_EVERY_N = 10000
DRIFT_WIDTH = 0
MAX_STREAM_SIZE = N_DRIFTS * (DRIFT_EVERY_N + DRIFT_WIDTH + 1)
WINDOW_MODE = 'WINDOW' if USE_WINDOW else 'POINT'
DRIFT_TYPE = 'ABRUPT' if DRIFT_WIDTH == 0 else 'GRADUAL'

stream_creator = CustomDriftStream(generator=GENERATOR,
                                   n_drifts=N_DRIFTS,
                                   drift_every_n=DRIFT_EVERY_N,
                                   drift_width=DRIFT_WIDTH)

stream = stream_creator.create_stream()
sch = stream.get_schema()

evaluator = ClassificationEvaluator(schema=sch, window_size=1)
learner = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])
student = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])

drifts = stream.get_drifts()
drifts = [(x.position, x.position + x.width) for x in drifts]

detector_perf = {}
for detector_name, detector in DETECTORS.items():
    print(f'Running detector: {detector_name}')

    np.random.seed(123)
    stream_creator = CustomDriftStream(generator=GENERATOR,
                                       n_drifts=N_DRIFTS,
                                       drift_every_n=DRIFT_EVERY_N,
                                       drift_width=DRIFT_WIDTH)

    stream = stream_creator.create_stream()

    sch = stream.get_schema()

    evaluator = ClassificationEvaluator(schema=sch, window_size=1)
    learner = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])
    student = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])

    drifts = stream.get_drifts()
    drifts = [(x.position, x.position + x.width) for x in drifts]

    detector_config = DETECTOR_SYNTH_PARAMS[GENERATOR][detector_name]

    if detector_name == 'STUDD':
        adwin_conf = DETECTOR_SYNTH_PARAMS[GENERATOR]['ADWIN']
        dct = DETECTORS['ADWIN'](**adwin_conf)

        detector_ = detector(student=student, detector=dct, **detector_config)
    else:
        detector_ = detector(**detector_config)

    wf = StreamingWorkflow(model=learner,
                           evaluator=evaluator,
                           detector=detector_,
                           use_window_perf=USE_WINDOW)

    wf.run_prequential(stream=stream, max_size=MAX_STREAM_SIZE)

    drift_eval = EvaluateDetector(max_delay=MAX_DELAY)

    metrics = drift_eval.calc_performance(trues=drifts,
                                          preds=wf.drift_predictions,
                                          tot_n_instances=wf.instances_processed)

    detector_perf[detector_name] = metrics

perf = pd.DataFrame(detector_perf).T

perf.to_csv(f'assets/results/{GENERATOR},{DRIFT_TYPE},{CLF},{WINDOW_MODE}.csv')
