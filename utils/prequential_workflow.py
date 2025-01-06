from typing import Optional


class StreamingWorkflow:
    """
    todo add adaptation mechanism
        retraining on buffer?

    """
    METRIC = 'accuracy'
    MIN_TRAINING_SIZE = 1000

    def __init__(self, model, evaluator, detector, use_window_perf: bool):
        self.model = model
        self.evaluator = evaluator
        self.detector = detector
        self.instances_processed = 0
        self.drift_predictions = []

        self.use_window_perf = use_window_perf

    def run_prequential(self, stream, max_size: Optional[int] = None):
        self._reset_params()

        while stream.has_more_instances():
            if self.instances_processed > max_size:
                break

            instance = stream.next_instance()
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

            self.model.train(instance)

            self.instances_processed += 1

    def _get_latest_score(self, true, pred):
        if self.use_window_perf:
            self.evaluator.update(true, pred)
            return self.evaluator.accuracy()
            # return self.evaluator.kappa()
        else:
            return int(true == pred)


    def _reset_params(self):
        self.instances_processed = 0
        self.drift_predictions = []
