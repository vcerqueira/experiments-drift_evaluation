import random
import numpy as np
import pandas as pd
import utils
import math

# add real drifts from capymoa with this

from capymoa.datasets import ElectricityTiny

stream = ElectricityTiny()
ElectricityTiny.
stream.extract()
stream.moa_stream
a=stream.next_instance()

stream.CLI_help()

stream.get_schema()
stream._length
stream.get_schema().get_num_attributes()

schema = stream.get_schema()
schema.get_label_indexes()

schema.get_num_attributes()

class DriftSimulator:
    # todo assumes x is numeric

    def __init__(self,
                 on_y_prior: bool,
                 on_x: bool,
                 drift_region=(0.3, 0.7),
                 label_skip_proba: float=0.75):

        self.on_y_prior = on_y_prior
        self.on_x = on_x
        self.drift_region = drift_region
        self.label_skip_proba = label_skip_proba

        self.fit = {}

    def fit(self, schema, stream_size):
        self.fit['drift_onset'] = int(stream_size * self.sample_drift_location())
        self.fit['selected_label'] = np.random.choice(schema.get_label_indexes(), 1)[0]
        self.fit['perm_idx'] = np.random.permutation(schema.get_num_attributes())

    def sample_drift_location(self):
        loc = np.random.uniform(self.drift_region[0], self.drift_region[1], 1)[0]

        return loc

    def _shuffle_arr(self, arr):
        shuffled_arr = np.take(arr, self.fit['perm_idx'])

        return shuffled_arr

