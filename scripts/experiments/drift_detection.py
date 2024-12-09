import numpy as np
import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator

from utils.streams import create_custom_drift_stream
from utils.evaluate import EvaluateDetector
from utils.prequential_workflow import StreamingWorkflow
from utils.config import CLASSIFIERS, DETECTORS

CLF = 'OnlineBagging'
MAX_DELAY = 500
MAX_STREAM_SIZE = 117000
USE_WINDOW = True

stream = create_custom_drift_stream(n_drifts=50,
                                    drift_every_n=2000,
                                    drift_width=250)

sch = stream.get_schema()
evaluator = ClassificationEvaluator(schema=sch, window_size=1)
learner = CLASSIFIERS[CLF](schema=sch)

drifts = stream.get_drifts()
drifts = [(x.position, x.position + x.width) for x in drifts]

detector_perf = {}
for detector_name, detector in DETECTORS.items():
    print(f'Running detector: {detector_name}')
    np.random.seed(123)
    wf = StreamingWorkflow(model=learner,
                           evaluator=evaluator,
                           detector=detector(),
                           use_window_perf=USE_WINDOW)

    wf.run_prequential(stream=stream, max_size=MAX_STREAM_SIZE)

    drift_eval = EvaluateDetector(max_delay=MAX_DELAY)

    metrics = drift_eval.calc_performance(trues=drifts,
                                          preds=wf.drift_predictions,
                                          tot_n_instances=wf.instances_processed)

    detector_perf[detector_name] = metrics

perf = pd.DataFrame(detector_perf).T

pd.set_option('display.max_columns', None)

# fix something weird about the results
perf.to_csv(f'assets/gradual,{CLF},window.csv')
