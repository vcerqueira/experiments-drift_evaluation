from capymoa.classifier import OnlineBagging
from capymoa.stream.generator import SEA
from capymoa.drift.detectors import ADWIN
from capymoa.stream.drift import AbruptDrift, GradualDrift, DriftStream
from capymoa.base import MOAClassifier
from typing_extensions import override
from capymoa.instance import LabeledInstance
from capymoa.stream import Schema
from utils.streams import create_custom_drift_stream

#


from capymoa.drift.base_detector import BaseDriftDetector, MOADriftDetector


class STUDD(BaseDriftDetector):

    def __init__(self,
                 min_n_instances: int,
                 # schema: Schema,
                 detector: MOADriftDetector,
                 student: MOAClassifier):
        super().__init__()

        self.min_n_instances = min_n_instances
        self.detector = detector
        self.student = student

        # self.schema = schema

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


stream = create_custom_drift_stream(n_drifts=50,
                                    drift_every_n=1000,
                                    drift_width=200)

model = OnlineBagging(schema=stream.get_schema(), ensemble_size=15)
st_model = OnlineBagging(schema=stream.get_schema(), ensemble_size=15)
# st_model.schema

# detector = ADWIN()
detector = STUDD(detector=ADWIN(),
                 min_n_instances=300,
                 student=st_model)

instances_processed = 0
max_size = 25000
MIN_TRAINING_SIZE = 500
drift_predictions = []

while stream.has_more_instances():
    if instances_processed > max_size:
        break

    instance = stream.next_instance()
    if instances_processed > MIN_TRAINING_SIZE:
        prediction = model.predict(instance)

        score = int(instance.y_index == prediction)

        # detector.add_element(score)
        detector.add_element(instance.x, prediction)
        if detector.detected_change():
            print(f'Change detected at index: {instances_processed}')
            drift_predictions.append(instances_processed)

    model.train(instance)

    instances_processed += 1
