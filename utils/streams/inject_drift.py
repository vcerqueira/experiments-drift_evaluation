from typing import Optional

import numpy as np
from capymoa.instance import LabeledInstance


class DriftSimulator:
    # todo assumes x is numeric

    # from capymoa.datasets import ElectricityTiny
    #
    # stream = ElectricityTiny()
    # schema = stream.get_schema()
    # instance = stream.next_instance()
    # drift_sim = DriftSimulator(schema=schema, on_y_prior=False, on_x=True)
    # drift_sim.fit(stream_size=stream._length)
    # t_instance = drift_sim.transform(instance)

    def __init__(self,
                 schema,
                 on_y_prior: bool,
                 on_x: bool,
                 drift_region=(0.3, 0.7),
                 burn_in_samples: int = 0,
                 label_skip_proba: float = 0.75):

        self.on_y_prior = on_y_prior
        self.on_x = on_x
        self.drift_region = drift_region
        self.burn_in_samples = burn_in_samples
        self.label_skip_proba = label_skip_proba
        self.schema = schema

        self.fitted = {}

    def fit(self, stream_size):
        drift_loc = self.sample_drift_location()
        assert drift_loc > 0
        if drift_loc < 1:
            self.fitted['drift_onset'] = int(stream_size * self.sample_drift_location())
        else:
            self.fitted['drift_onset'] = int(drift_loc)

        self.fitted['selected_label'] = np.random.choice(self.schema.get_label_indexes(), 1)[0]
        self.fitted['perm_idx'] = np.random.permutation(self.schema.get_num_attributes())

    def transform(self, instance) -> Optional[LabeledInstance]:
        if self.on_y_prior:
            # todo numa classe
            skip = self._skip_instance()
            if skip:
                return None

        if self.on_x:
            x_t = self._shuffle_arr(instance.x)
            t_instance = LabeledInstance.from_array(self.schema, x_t, instance.y_index)

            return t_instance

    def sample_drift_location(self):
        loc = np.random.uniform(self.drift_region[0] + self.burn_in_samples,
                                self.drift_region[1] - self.burn_in_samples, 1)[0]

        return loc

    def _shuffle_arr(self, arr):
        shuffled_arr = np.take(arr, self.fitted['perm_idx'])

        return shuffled_arr

    def _skip_instance(self):
        return np.random.binomial(1, 1 - self.label_skip_proba) < 1
