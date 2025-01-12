from typing import Optional

from utils.streams.inject_drift import DriftSimulator


class StreamingWorkflow:
    """
    todo add adaptation mechanism
        retraining on buffer?

    """
    METRIC = 'accuracy'
    MIN_TRAINING_SIZE = 1000

    def __init__(self,
                 model,
                 evaluator,
                 detector,
                 use_window_perf: bool,
                 drift_simulator: Optional[DriftSimulator] = None):

        self.model = model
        self.evaluator = evaluator
        self.detector = detector
        self.instances_processed = 0
        self.drift_predictions = []
        self.drift_simulator = drift_simulator

        self.use_window_perf = use_window_perf

    def run_prequential(self, stream, max_size: Optional[int] = None):
        self._reset_params()

        while stream.has_more_instances():
            if max_size is not None:
                if self.instances_processed > max_size:
                    break

            instance = stream.next_instance()

            if self.drift_simulator is not None:
                if self.instances_processed >= self.drift_simulator.fitted['drift_onset']:
                    instance = self.drift_simulator.transform(instance)

            if instance is None:
                self.instances_processed += 1
                continue

            if self.instances_processed > self.MIN_TRAINING_SIZE:
                prediction = self.model.predict(instance)

                score = self._get_latest_score(instance.y_index, prediction)

                if self.detector.__str__() == 'STUDD':
                    self.detector.add_element(instance.x, prediction)
                else:
                    self.detector.add_element(score)

                if self.detector.detected_change():
                    print(f'Change detected at index: {self.instances_processed}')
                    self.drift_predictions.append(self.instances_processed)

            if self.drift_simulator is not None:
                if self.instances_processed < self.drift_simulator.fitted['drift_onset']:
                    self.model.train(instance)
            else:
                self.model.train(instance)

            self.instances_processed += 1

    def _get_latest_score(self, true, pred):
        if self.use_window_perf:
            self.evaluator.update(true, pred)
            return self.evaluator.f1_score()
            # return self.evaluator.accuracy()
            # return self.evaluator.kappa()
        else:
            return int(true == pred)

    def _reset_params(self):
        self.instances_processed = 0
        self.drift_predictions = []
