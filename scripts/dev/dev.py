import numpy as np
import pandas as pd

import capymoa.drift.detectors as detectors

from utils.evaluate import EvaluateDetector

from capymoa.drift.detectors import ADWIN

detector = ADWIN(delta=0.001)

data_stream = np.random.randint(2, size=2000)
for i in range(999, 2000):
    data_stream[i] = np.random.randint(6, high=12)

for i in range(2000):
    detector.add_element(data_stream[i])
    if detector.detected_change():
        print(
            "Change detected in data: " + str(data_stream[i]) + " - at index: " + str(i)
        )

detector.detection_index

trues = np.array([1000])
preds = detector.detection_index
eval = EvaluateDetector(max_delay=200)
print(eval.calc_performance(preds, trues))



##



OB = OnlineBagging(schema=stream_sea2drift.get_schema(), ensemble_size=10)


# https://github.com/adaptive-machine-learning/CapyMOA/blob/main/src/capymoa/evaluation/evaluation.py#L888

