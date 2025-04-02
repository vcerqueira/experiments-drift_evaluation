import numpy as np
import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator

from utils.streams.inject_drift import DriftSimulator
from utils.evaluate import EvaluateDetector
from utils.prequential_workflow import StreamingWorkflow
from utils.streams.real import CAPYMOA_DATASETS, MAX_DELAY
from utils.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS

N_DRIFTS = 20

for drift_x in [False, True]:

    for drift_y in [False, True]:

        if np.sum([drift_x, drift_y]) < 1:
            continue

        DRIFT_TYPE = 'ABRUPT@XY' if (drift_x and drift_y) else ('ABRUPT@X' if drift_x
                                                                else 'ABRUPT@Y' if drift_y else None)

        for clf in [*CLASSIFIERS]:

            for ds in [*CAPYMOA_DATASETS]:
                print(ds)

                pre_stream = CAPYMOA_DATASETS[ds]()
                n = pre_stream._length

                pre_stream = DriftSimulator.shuffle_stream(pre_stream)
                x = pre_stream.next_instance()
                sch = pre_stream.get_schema()

                detector_perf = {}
                for detector_name, detector in DETECTORS.items():
                    print(f'Running detector: {detector_name}')

                    np.random.seed(123)

                    drift_episodes = []
                    for i in range(N_DRIFTS):
                        stream = CAPYMOA_DATASETS[ds]()
                        stream = DriftSimulator.shuffle_stream(stream)

                        drift_sim = DriftSimulator(on_x=drift_x,
                                                   on_y_prior=drift_y,
                                                   drift_region=(0.6, 0.9),
                                                   burn_in_samples=0,
                                                   schema=sch)

                        drift_sim.fit(stream_size=n)

                        drift_loc = drift_sim.fitted['drift_onset']
                        drifts = [(drift_loc, drift_loc)]

                        evaluator = ClassificationEvaluator(schema=sch, window_size=1)
                        learner = CLASSIFIERS[clf](schema=sch, **CLASSIFIER_PARAMS[clf])
                        student = CLASSIFIERS[clf](schema=sch, **CLASSIFIER_PARAMS[clf])

                        if detector_name == 'STUDD':
                            detector_ = detector(student=student)
                        else:
                            detector_ = detector()

                        wf = StreamingWorkflow(model=learner,
                                               evaluator=evaluator,
                                               detector=detector_,
                                               use_window_perf=False,
                                               min_training_size=int(n*0.5),
                                               start_detector_on_onset=False,
                                               drift_simulator=drift_sim)

                        wf.run_prequential(stream=stream)

                        drift_episodes.append({'preds': wf.drift_predictions, 'true': (drift_loc, drift_loc)})

                    drift_eval = EvaluateDetector(max_delay=MAX_DELAY[ds])

                    metrics = drift_eval.calc_performance_from_eps(drift_eps=drift_episodes, tot_n_instances=n)

                    detector_perf[detector_name] = metrics

                    perf = pd.DataFrame(detector_perf).T
                    perf.to_csv(f'assets/results/{ds},{DRIFT_TYPE},{clf}.csv')

                perf = pd.DataFrame(detector_perf).T
                perf.to_csv(f'assets/results/{ds},{DRIFT_TYPE},{clf}.csv')
