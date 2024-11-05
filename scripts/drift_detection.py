from pprint import pprint

from utils.streams import stream_sea2drift
from capymoa.drift.detectors import ADWIN
from capymoa.classifier import OnlineBagging
from capymoa.evaluation.evaluation import ClassificationEvaluator

METRIC = 'accuracy'
MIN_TRAINING_SIZE = 1000
MAX_SIZE = 15000

sch = stream_sea2drift.get_schema()
detector = ADWIN(delta=0.001)

evaluator = ClassificationEvaluator(schema=sch, window_size=1)
learner = OnlineBagging(schema=sch, ensemble_size=10)

instances_processed = 0
drift_predictions = []
while stream_sea2drift.has_more_instances():
    if instances_processed > MAX_SIZE:
        break

    instance = stream_sea2drift.next_instance()

    if instances_processed > MIN_TRAINING_SIZE:
        prediction = learner.predict(instance)

        evaluator.update(instance.y_index, prediction)

        score = evaluator.accuracy()

        detector.add_element(score)
        if detector.detected_change():
            print(f'Change detected at index: {instances_processed}')
            drift_predictions.append(instances_processed)

    learner.train(instance)

    instances_processed += 1

# evaluator.metrics_per_window()
# evaluator.accuracy()

# pprint(evaluator.metrics_dict())
