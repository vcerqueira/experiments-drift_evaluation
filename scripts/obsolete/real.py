import numpy as np
import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator

from utils.streams.inject_drift import DriftSimulator
from utils.evaluate import EvaluateDetector
from utils.prequential_workflow import StreamingWorkflow
from utils.streams.real import CAPYMOA_DATASETS, STABLE_PERIODS
from utils.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS

# CLF = 'ARF'
CLF = 'HoeffdingTree'
DATASET = 'Electricity'
USE_WINDOW = False
# MAX_DELAY_PERC = 0.1
N_DRIFTS = 30
DRIFT_ON_X = False
DRIFT_ON_Y = True
WINDOW_MODE = 'WINDOW' if USE_WINDOW else 'POINT'
DRIFT_TYPE = 'ABRUPT@X' if DRIFT_ON_X else 'ABRUPT@Y'

stream = CAPYMOA_DATASETS[DATASET]()
n = stream._length
stream = DriftSimulator.shuffle_stream(stream)
x = stream.next_instance()
# print(x.x)
sch = stream.get_schema()

# max_delay = int(n * MAX_DELAY_PERC)
max_delay = 1000

detector_perf = {}
for detector_name, detector in DETECTORS.items():
    print(f'Running detector: {detector_name}')
    # if detector_name != 'STUDD':
    #     continue

    drift_episodes = []
    for i in range(N_DRIFTS):
        stream = CAPYMOA_DATASETS[DATASET]()

        stb_period_idx = np.random.choice(range(len(STABLE_PERIODS[DATASET])), 1)
        stb_period = STABLE_PERIODS[DATASET][stb_period_idx[0]]

        drift_sim = DriftSimulator(on_x=DRIFT_ON_X,
                                   on_y_prior=DRIFT_ON_Y,
                                   drift_region = stb_period,
                                   burn_in_samples=max_delay,
                                   schema=sch)

        drift_sim.fit(stream_size=n)

        drift_loc = drift_sim.fitted['drift_onset']
        drifts = [(drift_loc, drift_loc)]

        evaluator = ClassificationEvaluator(schema=sch, window_size=1)
        learner = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])
        student = CLASSIFIERS[CLF](schema=sch, **CLASSIFIER_PARAMS[CLF])

        if detector_name == 'STUDD':
            detector_ = detector(student=student)
        else:
            detector_ = detector()

        wf = StreamingWorkflow(model=learner,
                               evaluator=evaluator,
                               detector=detector_,
                               use_window_perf=USE_WINDOW,
                               start_detector_on_onset=False,
                               drift_simulator=drift_sim)

        wf.run_prequential(stream=stream, max_size=stb_period[1]+1)

        alarms = [x for x in wf.drift_predictions if x >= stb_period[0]]

        drift_episodes.append({'preds': alarms, 'true': (drift_loc, drift_loc)})

    drift_eval = EvaluateDetector(max_delay=max_delay)

    metrics = drift_eval.calc_performance_from_eps(drift_eps=drift_episodes, tot_n_instances=n)

    detector_perf[detector_name] = metrics

    perf = pd.DataFrame(detector_perf).T
    perf.to_csv(f'assets/results/{DATASET},{DRIFT_TYPE},{CLF},{WINDOW_MODE}.csv')

perf = pd.DataFrame(detector_perf).T
perf.to_csv(f'assets/results/{DATASET},{DRIFT_TYPE},{CLF},{WINDOW_MODE}.csv')
