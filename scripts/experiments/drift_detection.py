import numpy as np
import pandas as pd
from capymoa.evaluation.evaluation import ClassificationEvaluator

from utils.streams import stream_sea_abrupt as stream
from utils.evaluate import EvaluateDetector
from utils.prequential_workflow import StreamingWorkflow
from utils.config import MAX_STREAM_SIZE, CLASSIFIERS, DETECTORS

CLF = 'OnlineBagging'
MAX_DELAY = 500

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
                           use_window_perf=True)

    wf.run_prequential(stream=stream, max_size=MAX_STREAM_SIZE)

    drift_eval = EvaluateDetector(max_delay=MAX_DELAY)

    metrics = drift_eval.calc_performance(trues=drifts,
                                          preds=wf.drift_predictions,
                                          tot_n_instances=wf.instances_processed)

    detector_perf[detector_name] = metrics

perf = pd.DataFrame(detector_perf).T

pd.set_option('display.max_columns', None)

# fix something weird about the results
perf
