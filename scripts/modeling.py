from pprint import pprint
import numpy as np
import pandas as pd

import capymoa.drift.detectors as detectors

from capymoa.classifier import OnlineBagging

from utils.streams import stream_sea2drift

from capymoa.evaluation.evaluation import ClassificationEvaluator

##
METRIC = 'accuracy'
MIN_TRAINING_SIZE = 1000
MAX_SIZE = 15000

sch = stream_sea2drift.get_schema()

# stream_sea2drift.get_schema()
# stream_sea2drift.next_instance()

evaluator = ClassificationEvaluator(schema=sch, window_size=1)

learner = OnlineBagging(schema=sch, ensemble_size=10)

# https://github.com/adaptive-machine-learning/CapyMOA/blob/main/src/capymoa/evaluation/evaluation.py#L888


error_rate = []
instances_processed = 0
while stream_sea2drift.has_more_instances():
    if instances_processed > MAX_SIZE:
        break

    instance = stream_sea2drift.next_instance()

    if instances_processed > MIN_TRAINING_SIZE:
        prediction = learner.predict(instance)

        evaluator.update(instance.y_index, prediction)

    learner.train(instance)

    instances_processed += 1

evaluator.metrics_per_window()
evaluator.accuracy()

pprint(evaluator.metrics_dict())
