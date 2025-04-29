from pathlib import Path

import numpy as np

from capymoa.drift.detectors import ABCD

from src.streams.synth import CustomDriftStream

MAX_DELAY = 1000
N_DRIFTS = 50
DRIFT_EVERY_N = 2000
DRIFT_WIDTH = 0
DRIFT_TYPE = 'ABRUPT' if DRIFT_WIDTH == 0 else 'GRADUAL'
MAX_STREAM_SIZE = N_DRIFTS * (DRIFT_EVERY_N + DRIFT_WIDTH + 1)
RANDOM_SEED = 123
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / 'assets' / 'results'

np.random.seed(RANDOM_SEED)
stream_creator = CustomDriftStream(
    generator='STAGGER',
    n_drifts=N_DRIFTS,
    drift_every_n=DRIFT_EVERY_N,
    drift_width=DRIFT_WIDTH
)

stream = stream_creator.create_stream()

params = {'delta_drift': 0.6,
          'delta_warn': 0.01,
          'encoding_factor': 0.3,
          'model_id': 'ae',
          'num_splits': 50}

# detector_instance = ABCD(**params)
detector_instance = ABCD()

instances_processed = 0
drift_predictions = []
while stream.has_more_instances():
    if instances_processed > 100000:
        break

    instance = stream.next_instance()

    detector_instance.add_element(instance)

    if detector_instance.detected_change():
        print(f'Change detected at index: {instances_processed}')
        drift_predictions.append(instances_processed)

    instances_processed += 1

print(drift_predictions)

# ----- from the tutorials


from capymoa.drift.detectors import ABCD
from capymoa.datasets import ElectricityTiny

detector = ABCD()
# detector = ABCD(model_id="pca")

## Opening a file as a stream
stream = ElectricityTiny()
i = 0
while stream.has_more_instances and i < 5000:
    i += 1
    instance = stream.next_instance()
    detector.add_element(instance)
    if detector.detected_change():
        print("Change detected at index: " + str(i))
