from pprint import pprint
import numpy as np
import pandas as pd

import capymoa.drift.detectors as detectors

from capymoa.classifier import OnlineBagging

from utils.streams import stream_sea2drift

from capymoa.evaluation.evaluation import ClassificationEvaluator

##

# stream_sea2drift.get_schema()
# stream_sea2drift.next_instance()

# nesta fase nao preciso disto... posso usar error rate simples
evaluator_cumulative = ClassificationEvaluator(
    schema=stream_sea2drift.get_schema(), window_size=1
)

learner = OnlineBagging(schema=stream_sea2drift.get_schema(), ensemble_size=10)

# https://github.com/adaptive-machine-learning/CapyMOA/blob/main/src/capymoa/evaluation/evaluation.py#L888

MIN_TRAINING_SIZE = 100

error_rate = []
instances_processed = 0
while stream_sea2drift.has_more_instances():
    if instances_processed > 1500:
        break

    instance = stream_sea2drift.next_instance()

    if instances_processed > MIN_TRAINING_SIZE:
        prediction = learner.predict(instance)

        y = instance.y_index

        error_rate.append(y == prediction)
        evaluator_cumulative.update(y, prediction)

    learner.train(instance)

    instances_processed += 1

evaluator_cumulative.metrics_per_window()

pprint(evaluator_cumulative.metrics_dict())