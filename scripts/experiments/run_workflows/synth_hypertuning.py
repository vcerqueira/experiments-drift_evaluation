import numpy as np
import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator
from sklearn.model_selection import ParameterSampler

from utils.streams.synth import CustomDriftStream
from utils.evaluate import EvaluateDetector
from utils.prequential_workflow import StreamingWorkflow
from utils.config import CLASSIFIERS, DETECTORS, CLASSIFIER_PARAMS, DETECTOR_PARAM_SPACE

"""
In this script, we optimize a detector's parameters using a grid search

for each detector:
sample 50 configurations
store results


"""

# CLF = 'NaiveBayes'
USE_WINDOW = False
MAX_DELAY = 1000
N_DRIFTS = 30
DRIFT_EVERY_N = 10000
DRIFT_WIDTH = 0
MAX_STREAM_SIZE = N_DRIFTS * (DRIFT_EVERY_N + DRIFT_WIDTH + 1)
WINDOW_MODE = 'WINDOW' if USE_WINDOW else 'POINT'
DRIFT_TYPE = 'ABRUPT' if DRIFT_WIDTH == 0 else 'GRADUAL'
N_ITER_RANDOM_SEARCH = 30

performance_metrics = []
for detector_name, detector in DETECTORS.items():
    print(f'Running detector: {detector_name}')

    if detector_name not in ['ABCDx']:
        continue

    config_space = ParameterSampler(param_distributions=DETECTOR_PARAM_SPACE[detector_name],
                                    n_iter=N_ITER_RANDOM_SEARCH)

    for config_ in config_space:
        print(config_)

        for clf in [*CLASSIFIERS]:

            for generator in [*CustomDriftStream.GENERATORS]:
                # generator

                np.random.seed(123)
                stream_creator = CustomDriftStream(generator=generator,
                                                   n_drifts=N_DRIFTS,
                                                   drift_every_n=DRIFT_EVERY_N,
                                                   drift_width=DRIFT_WIDTH)

                stream = stream_creator.create_stream()
                sch = stream.get_schema()

                evaluator = ClassificationEvaluator(schema=sch, window_size=1)
                learner = CLASSIFIERS[clf](schema=sch, **CLASSIFIER_PARAMS[clf])
                student = CLASSIFIERS[clf](schema=sch, **CLASSIFIER_PARAMS[clf])

                drifts = stream.get_drifts()
                drifts = [(x.position, x.position + x.width) for x in drifts]

                if detector_name == 'STUDD':
                    detector_ = detector(student=student, **config_)
                else:
                    detector_ = detector(**config_)

                wf = StreamingWorkflow(model=learner,
                                       evaluator=evaluator,
                                       detector=detector_,
                                       use_window_perf=USE_WINDOW)

                monitor_x = True if detector_name == 'ABCDx' else False

                wf.run_prequential(stream=stream,
                                   max_size=MAX_STREAM_SIZE,
                                   monitor_instance=monitor_x)

                drift_eval = EvaluateDetector(max_delay=MAX_DELAY)

                metrics = drift_eval.calc_performance(trues=drifts,
                                                      preds=wf.drift_predictions,
                                                      tot_n_instances=wf.instances_processed)

                metadata = {
                    'detector': detector_name, 'stream': generator,
                    'learner': clf, 'drift_type': DRIFT_TYPE,
                }

                results = {**metadata, 'params': config_, **metrics}

                performance_metrics.append(results)

                perf = pd.DataFrame(performance_metrics)

                perf.to_csv(f'assets/results/detector_hypertuning3,{DRIFT_TYPE}.csv', index=False)

perf = pd.DataFrame(performance_metrics)

perf.to_csv(f'assets/results/detector_hypertuning3,{DRIFT_TYPE}.csv', index=False)
