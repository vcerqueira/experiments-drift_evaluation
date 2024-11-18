from typing import Optional
from pprint import pprint

from utils.streams import stream_sea2drift
from capymoa.drift.detectors import ADWIN, PageHinkley
from capymoa.classifier import OnlineBagging
from capymoa.evaluation.evaluation import ClassificationEvaluator
from utils.evaluate import EvaluateDetector

MAX_SIZE = 15000

sch = stream_sea2drift.get_schema()
evaluator = ClassificationEvaluator(schema=sch, window_size=1)
learner = OnlineBagging(schema=sch, ensemble_size=10)

stream_sea2drift.drifts[1].position
stream_sea2drift.drifts[1].width

drifts = stream_sea2drift.get_drifts()
drifts = [(x.position, x.position + x.width) for x in drifts]


class StreamingWorkflow:
    """
    todo add adaptation mechanism
        retraining on buffer?

    """
    METRIC = 'accuracy'
    MIN_TRAINING_SIZE = 1000

    def __init__(self, model, evaluator, detector):
        self.model = model
        self.evaluator = evaluator
        self.detector = detector
        self.instances_processed = 0
        self.drift_predictions = []

    def run_prequential(self, stream, max_size: Optional[int] = None):
        self._reset_params()

        while stream.has_more_instances():
            if self.instances_processed > max_size:
                break

            instance = stream.next_instance()
            if self.instances_processed > self.MIN_TRAINING_SIZE:
                prediction = self.model.predict(instance)

                self.evaluator.update(instance.y_index, prediction)

                score = self._get_latest_score()

                self._update_detector(score)

            self.model.train(instance)

            self.instances_processed += 1

    def _get_latest_score(self):
        return self.evaluator.accuracy()

    def _update_detector(self, score):
        self.detector.add_element(score)
        if self.detector.detected_change():
            print(f'Change detected at index: {self.instances_processed}')
            self.drift_predictions.append(self.instances_processed)

    def _reset_params(self):
        self.instances_processed = 0
        self.drift_predictions = []


# detector = ADWIN(delta=0.001)
detector = PageHinkley()

wf = StreamingWorkflow(model=learner,
                       evaluator=evaluator,
                       detector=detector)

wf.run_prequential(stream=stream_sea2drift, max_size=MAX_SIZE)
wf.drift_predictions

drift_eval = EvaluateDetector(max_delay=500)
eps = drift_eval._get_drift_episodes(preds=wf.drift_predictions, trues=drifts)

for ep in eps:
    print(ep)
    drift_detected = False
    min_detection_time = float('inf')

    for pred in ep['preds']:
        if ep['true'] - max_delay <= pred <= episode.true + self.max_delay:
            drift_detected = True
            detection_time = pred - episode.true
            min_detection_time = min(min_detection_time, detection_time)
        else:
            fp += 1

    if drift_detected:
        tp += 1
        detection_times.append(min_detection_time)
    else:
        fn += 1


