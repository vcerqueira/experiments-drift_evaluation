from typing import Optional

import numpy as np

from src.streams.inject_drift import DriftSimulator
from src.config import CLASSIFIERS, CLASSIFIER_PARAMS, DETECTORS


class SupervisedStreamingWorkflow:

    def __init__(self,
                 model,
                 evaluator,
                 detector,
                 use_window_perf: bool,
                 min_training_size: int = 1000,
                 # start_detector_on_onset: bool = False,
                 drift_simulator: Optional[DriftSimulator] = None):

        self.model = model
        self.evaluator = evaluator
        self.detector = detector
        self.instances_processed = 0
        self.min_training_size = min_training_size
        self.drift_predictions = []
        self.drift_simulator = drift_simulator
        # self.start_detector_on_onset = start_detector_on_onset

        self.use_window_perf = use_window_perf

    def run_prequential(self,
                        stream,
                        max_size: Optional[int] = None,
                        monitor_instance: bool = False):
        self._reset_params()

        while stream.has_more_instances():
            if max_size is not None:
                if self.instances_processed > max_size:
                    break

            instance = stream.next_instance()

            if self.drift_simulator is not None:
                if self.drift_simulator.apply_drift(self.instances_processed):
                    instance = self.drift_simulator.transform(instance)

            if instance is None:
                self.instances_processed += 1
                continue

            if self.instances_processed > self.min_training_size:
                prediction = self.model.predict(instance)

                # todo where is this relevant?
                # if self.start_detector_on_onset:
                #     if self.instances_processed < self.drift_simulator.fitted['drift_onset']:
                #         continue

                score = self._get_latest_score(instance.y_index, prediction)

                if monitor_instance:
                    self.detector.add_element(instance)
                else:
                    if self.detector.__str__() == 'STUDD':
                        self.detector.add_element(instance, prediction)
                    else:
                        try:
                            # self.detector.add_element(float(score))
                            self.detector.add_element(score)
                        except TypeError:
                            self.detector.add_element(float(score))

                if self.detector.detected_change():
                    print(f'Change detected at index: {self.instances_processed}')
                    self.drift_predictions.append(self.instances_processed)

            # if self.drift_simulator is not None:
            #     if self.instances_processed < self.drift_simulator.fitted['drift_onset']:
            #         self.model.train(instance)
            # else:
            #     self.model.train(instance)
            # continue training the model
            self.model.train(instance)

            self.instances_processed += 1

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


class ProxyStreamingWorkflow:

    def __init__(self,
                 schema,
                 learner: str,
                 detector_name: str,
                 evaluator,
                 evaluator_detection,
                 use_window_perf: bool,
                 update_after_alarm: bool = False,
                 verification_delay: int = 0,
                 supervision_proba: float = 1.0,
                 buffer_size: int = 1000,
                 min_training_size: int = 1000):

        self.schema = schema
        self.learner = learner
        self.model = CLASSIFIERS[self.learner](schema=self.schema, **CLASSIFIER_PARAMS[self.learner])
        self.evaluator = evaluator
        self.evaluator_detection = evaluator_detection
        self.detector_name = detector_name
        if self.detector_name == 'STUDD':
            std = CLASSIFIERS[self.learner](schema=self.schema, **CLASSIFIER_PARAMS[self.learner])
            self.detector = DETECTORS[self.detector_name](student=std)
        else:
            self.detector = DETECTORS[self.detector_name]()

        self.min_training_size = min_training_size
        self.verification_delay = verification_delay
        self.supervision_proba = supervision_proba
        self.buffer_size = buffer_size
        self.use_window_perf = use_window_perf
        self.update_after_alarm = update_after_alarm
        self.instances_processed = 0
        self.instances_since_retrain = 0
        self.drift_predictions = []
        self.instance_delayed_buffer = []
        self.instance_adaptation_buffer = []

    def run_prequential(self,
                        stream,
                        max_size: Optional[int] = None,
                        monitor_instance: bool = False):

        self._reset_params()

        while stream.has_more_instances():
            if max_size is not None:
                if self.instances_processed > max_size:
                    break

            instance = stream.next_instance()

            prediction = self.model.predict(instance)
            self.evaluator.update(instance.y_index, prediction)

            if self.instances_processed > self.min_training_size:

                if self.instances_since_retrain > self.min_training_size:
                    # studd is not being trained
                    if self.detector.__str__() == 'STUDD' or monitor_instance:
                        self.__add_instance_to_detector(instance, prediction, monitor_instance)

                self.instance_delayed_buffer.append((instance, prediction))

                if len(self.instance_delayed_buffer) > self.verification_delay:
                    delayed_instance, delayed_prediction = self.instance_delayed_buffer.pop(0)

                    instance_available = np.random.binomial(1, self.supervision_proba)

                    if instance_available:
                        self.instance_adaptation_buffer.append(delayed_instance)
                        if len(self.instance_adaptation_buffer) > self.buffer_size:
                            self.instance_adaptation_buffer.pop(0)

                        # Only add to detector if it's not STUDD and we're not monitoring instances
                        # This is because STUDD and monitor_instance cases were already handled earlier
                        if self.instances_since_retrain > self.min_training_size:
                            if not (self.detector.__str__() == 'STUDD' or monitor_instance):
                                self.__add_instance_to_detector(delayed_instance,
                                                                delayed_prediction,
                                                                monitor_instance)

                        self.model.train(delayed_instance)

                if self.instances_since_retrain > self.min_training_size:
                    if self.detector.detected_change():
                        self.instances_since_retrain = 0
                        print(f'Change detected at index: {self.instances_processed}')
                        self.drift_predictions.append(self.instances_processed)

                        if self.update_after_alarm:
                            # reset model and detector
                            self.model = CLASSIFIERS[self.learner](schema=self.schema,
                                                                   **CLASSIFIER_PARAMS[self.learner])
                            if self.detector_name == 'STUDD':
                                std = CLASSIFIERS[self.learner](schema=self.schema, **CLASSIFIER_PARAMS[self.learner])
                                self.detector = DETECTORS[self.detector_name](student=std)
                            else:
                                self.detector = DETECTORS[self.detector_name]()

                            # train on adaptation buffer
                            # ... is studd's meta being fit??
                            for instance in self.instance_adaptation_buffer:
                                self.model.train(instance)
            else:
                self.model.train(instance)

            self.instances_processed += 1
            self.instances_since_retrain += 1

    def __add_instance_to_detector(self, instance, prediction, monitor_instance):
        score = self._get_latest_score(instance.y_index, prediction)

        if monitor_instance:
            self.detector.add_element(instance)
        else:
            if self.detector.__str__() == 'STUDD':
                self.detector.add_element(instance, prediction)
            else:
                self.detector.add_element(score)

    def _get_latest_score(self, true, pred):
        if self.use_window_perf:
            self.evaluator_detection.update(true, pred)
            return self.evaluator_detection.f1_score()
            # return self.evaluator.accuracy()
            # return self.evaluator.kappa()
        else:
            self.evaluator_detection.update(true, pred)
            return int(true == pred)

    def _reset_params(self):
        self.instances_processed = 0
        self.drift_predictions = []
