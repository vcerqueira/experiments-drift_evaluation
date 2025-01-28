from typing import Optional, Tuple

from utils.streams.inject_drift import DriftSimulator


class StreamingWorkflow:
    MIN_TRAINING_SIZE = 1000

    def __init__(self,
                 model,
                 evaluator,
                 detector,
                 use_window_perf: bool,
                 start_detector_on_onset: bool = False,
                 drift_simulator: Optional[DriftSimulator] = None):

        self.model = model
        self.evaluator = evaluator
        self.detector = detector
        self.instances_processed = 0
        self.drift_predictions = []
        self.drift_simulator = drift_simulator
        self.start_detector_on_onset = start_detector_on_onset

        self.use_window_perf = use_window_perf

    def run_prequential(self,
                        stream,
                        max_size: Optional[int] = None):
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

                if self.start_detector_on_onset:
                    if self.instances_processed < self.drift_simulator.fitted['drift_onset'] - 10000:
                        continue

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

    def run_prequential_ensemble_detectors(self, stream, max_size: Optional[int] = None):
        self._reset_params()

        assert isinstance(self.detector, dict), \
            'self.detector should be a dict containing an ensemble of detectors'

        detectors_alarms = {k: [] for k in [*self.detector]}

        while stream.has_more_instances():
            if max_size is not None:
                if self.instances_processed > max_size:
                    break

            instance = stream.next_instance()

            if self.instances_processed > self.MIN_TRAINING_SIZE:
                prediction = self.model.predict(instance)

                score = self._get_latest_score(instance.y_index, prediction)

                for detector_name, detector in self.detector.items():

                    if detector_name == 'STUDD':
                        detector.add_element(instance.x, prediction)
                    else:
                        detector.add_element(score)

                    if detector.detected_change():
                        detectors_alarms[detector_name].append(self.instances_processed)

            self.model.train(instance)

            self.instances_processed += 1

        return detectors_alarms

    def _get_latest_score(self, true, pred):
        if self.use_window_perf:
            self.evaluator.update(true, pred)
            # return self.evaluator.f1_score()
            return self.evaluator.accuracy()
            # return self.evaluator.kappa()
        else:
            return int(true == pred)

    def _reset_params(self):
        self.instances_processed = 0
        self.drift_predictions = []

    @staticmethod
    def find_stable_periods(alarms_dict, min_gap: int, n: int, max_delay: int):
        all_alarms = []
        for alarm_ in alarms_dict.values():
            all_alarms.extend(alarm_)

        all_alarms = sorted(set(all_alarms))

        if not all_alarms:
            return []

        stable_periods = []
        for i in range(len(all_alarms) - 1):
            gap = all_alarms[i + 1] - all_alarms[i]
            period_end = all_alarms[i + 1]

            if gap >= min_gap and (n - period_end) >= max_delay:
                stable_periods.append((all_alarms[i], period_end))

        return stable_periods
