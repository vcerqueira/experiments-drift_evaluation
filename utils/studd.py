from typing_extensions import override
from capymoa.base import MOAClassifier
from capymoa.drift.detectors import ADWIN
from capymoa.drift.base_detector import BaseDriftDetector, MOADriftDetector
from capymoa.instance import LabeledInstance
from capymoa.classifier import AdaptiveRandomForestClassifier as ARF


class STUDD(BaseDriftDetector):

    def __init__(self,
                 student: MOAClassifier,
                 min_n_instances: int = 500,
                 detector: MOADriftDetector = ADWIN()):
        super().__init__()

        self.min_n_instances = min_n_instances
        self.detector = detector
        self.student = student

        self.in_concept_change = self.detector.in_concept_change
        self.in_warning_zone = self.detector.in_warning_zone
        self.detection_index = self.detector.detection_index
        self.warning_index = self.detector.warning_index
        self.data = self.detector.data

    def __str__(self):
        return 'STUDD'

    @override
    def add_element(self, instance_x, teacher_prediction) -> None:
        tmp_instance = self.instance_from_instance(instance_x, teacher_prediction)

        if self.idx >= self.min_n_instances:
            pred = self.student.predict(tmp_instance)

            meta_y = int(pred == teacher_prediction)

            self.detector.add_element(meta_y)
            self.in_concept_change = self.detector.in_concept_change
            self.in_warning_zone = self.detector.in_warning_zone
            self.detection_index = self.detector.detection_index
            self.warning_index = self.detector.warning_index
            self.data = self.detector.data

        self.student.train(tmp_instance)

        self.idx += 1

    def instance_from_instance(self, x, y):
        tmp_instance = LabeledInstance.from_array(self.student.schema, x, y)

        return tmp_instance

    @override
    def get_params(self):
        return self.detector.get_params()

    @override
    def reset(self, clean_history: bool = False) -> None:
        """Reset the drift detector.

        :param clean_history: Whether to reset detection history, defaults to False
        """
        self.in_concept_change = False
        self.in_warning_zone = False

        if clean_history:
            self.detection_index = []
            self.warning_index = []
            self.data = []
            self.idx = 0
            self.detector.reset(clean_history)
