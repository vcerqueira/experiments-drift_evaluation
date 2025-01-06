from capymoa.classifier import OnlineBagging
from capymoa.stream.generator import SEA
from capymoa.drift.detectors import ADWIN
from utils.streams import create_custom_drift_stream

from utils.studd import STUDD

#

stream = create_custom_drift_stream(n_drifts=50,
                                    drift_every_n=1000,
                                    drift_width=200)

model = OnlineBagging(schema=stream.get_schema(), ensemble_size=15)
st_model = OnlineBagging(schema=stream.get_schema(), ensemble_size=15)
# st_model.schema

# detector = ADWIN()
detector = STUDD(student=st_model)

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
