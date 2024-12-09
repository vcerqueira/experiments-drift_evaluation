from capymoa.classifier import OnlineBagging
from capymoa.stream.generator import SEA
from capymoa.stream import Stream
from capymoa.stream.drift import AbruptDrift, GradualDrift, DriftStream
from capymoa.base import MOAClassifier
from typing_extensions import override

from xgboost import XGBRFRegressor

rf = XGBRFRegressor()
rf.fit()

from moa.classifiers.meta import AdaptiveRandomForest as _MOA_AdaptiveRandomForest
from moa.classifiers.core.driftdetection import ChangeDetector
from moa.classifiers.core.driftdetection import ADWINChangeDetector

from capymoa.drift.base_detector import BaseDriftDetector, MOADriftDetector

stream_sea2drift = DriftStream(stream=[SEA(function=1),
                                       AbruptDrift(position=5000),
                                       SEA(function=3),
                                       GradualDrift(start=9000, end=12000),
                                       SEA(function=1)])

sample = stream_sea2drift.next_instance()
sample.x
sample.y_label

# while stream_sea2drift.has_more_instances():


model = OnlineBagging(schema=stream_sea2drift.get_schema(), ensemble_size=5)


class STUDD(BaseDriftDetector):

    def __init__(self,
                 min_n_instances: int,
                 detector: MOADriftDetector,
                 student: MOAClassifier):
        super().__init__()

        self.min_n_instances = min_n_instances
        self.detector = detector
        self.student = student

    def __str__(self):
        return 'STUDD'

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

    @override
    def add_element(self, instance, teacher_prediction) -> None:
        self.student.train([instance.x], [teacher_prediction])

        self.moa_detector.input(element)
        self.data.append(element)
        self.idx += 1

        self.in_concept_change = self.moa_detector.getChange()
        self.in_warning_zone = self.moa_detector.getWarningZone()

        if self.in_warning_zone:
            self.warning_index.append(self.idx)

        if self.in_concept_change:
            self.detection_index.append(self.idx)

    @override
    def get_params(self) -> Dict[str, Any]:
        options = list(self.moa_detector.getOptions().getOptionArray())
        return {opt.getName(): opt.getValueAsCLIString() for opt in options}
