import random
from typing import Optional

import numpy as np
import pandas as pd
from capymoa.instance import LabeledInstance
from capymoa.stream import NumpyStream

from utils.streams.real import STREAM_MEDIANS


class DriftSimulator:
    """
    A class for simulating concept drift in data streams by introducing controlled changes to the data.
    
    This class provides functionality to inject two types of drift:
    1. Label-based drift: Randomly skips instances of a selected label with a given probability
    2. Feature-based drift: Shuffles the order of features in the instances
    
    The drift is introduced at a random location within a specified region of the stream.
    
    Attributes:
        schema: The schema of the data stream
        on_y_prior (bool): Whether to apply label-based drift
        on_x (bool): Whether to apply feature-based drift
        drift_region (tuple): The region (as proportion of stream length) where drift can occur
        burn_in_samples (int): Number of samples to skip at the beginning of the stream
        label_skip_proba (float): Probability of skipping an instance when label-based drift is active
        fitted (dict): Stores the fitted parameters after calling fit()
    
    Example:
        >>> from capymoa.datasets import ElectricityTiny
        >>> stream = ElectricityTiny()
        >>> schema = stream.get_schema()
        >>> drift_sim = DriftSimulator(schema=schema, on_y_prior=False, on_x=True)
        >>> drift_sim.fit(stream_size=stream._length)
        >>> instance = stream.next_instance()
        >>> t_instance = drift_sim.transform(instance)
    """

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
                 on_y_swap: bool,
                 on_x_permute: bool,
                 on_x_exceed: bool,
                 drift_region=(0.3, 0.7),
                 burn_in_samples: int = 0,
                 label_skip_proba: float = 0.75):
        """
        Initialize the DriftSimulator.
        
        Args:
            schema: The schema of the data stream
            on_y_prior (bool): Whether to apply label-based drift
            on_x (bool): Whether to apply feature-based drift
            drift_region (tuple): The region (as proportion of stream length) where drift can occur
            burn_in_samples (int): Number of samples to skip at the beginning of the stream
            label_skip_proba (float): Probability of skipping an instance when label-based drift is active
        """
        self.on_y_prior = on_y_prior
        self.on_y_swap = on_y_swap
        self.on_x_permute = on_x_permute
        self.on_x_exceed = on_x_exceed
        self.drift_region = drift_region
        self.burn_in_samples = burn_in_samples
        self.label_skip_proba = label_skip_proba
        self.schema = schema

        self.fitted = {}

    def fit(self, stream_size):
        """
        Fit the drift simulator to the stream.
        
        This method:
        1. Samples a random location for the drift onset
        2. Selects a random label for label-based drift
        3. Generates a random permutation of feature indices for feature-based drift
        
        Args:
            stream_size (int): The total size of the stream
        """
        drift_loc = self.sample_drift_location()
        assert drift_loc > 0
        if drift_loc < 1:
            self.fitted['drift_onset'] = int(stream_size * self.sample_drift_location())
        else:
            self.fitted['drift_onset'] = int(drift_loc)

        self.fitted['y_selected_label'] = np.random.choice(self.schema.get_label_indexes(), 1)[0]
        self.fitted['y_selected_swap'] = np.random.choice([x for x in self.schema.get_label_indexes() if
                                                           x != self.fitted['y_selected_label']], 1)[0]
        self.fitted['x_perm_idx'] = np.random.permutation(self.schema.get_num_attributes())
        self.fitted['x_exceed_attr'], self.fitted['x_exceed_idx'] = self.select_random_num_attr(self.schema)
        self.fitted['x_exceed_val'] = STREAM_MEDIANS[self.schema.dataset_name][self.fitted['x_exceed_attr']]

    def transform(self, instance) -> Optional[LabeledInstance]:
        """
        Transform a single instance by applying the configured drift.
        
        Args:
            instance (LabeledInstance): The instance to transform
            
        Returns:
            Optional[LabeledInstance]: The transformed instance, or None if the instance was skipped
        """
        # instance_ = copy.deepcopy(instance)

        if self.on_y_prior:
            if instance.y_index == self.fitted['y_selected_label']:
                skip = self._skip_instance()
                if skip:
                    return None

        if self.on_y_swap:
            if instance.y_index == self.fitted['y_selected_label']:
                y_swapper = self.fitted['y_selected_swap']

                instance = LabeledInstance.from_array(self.schema,
                                                      instance.x,
                                                      y_swapper)

        if self.on_x_permute:
            x_t = self._shuffle_arr(instance.x)
            instance = LabeledInstance.from_array(self.schema, x_t, instance.y_index)

        if self.on_x_exceed:
            exceeds_threshold = self._arr_exceeds_threshold(instance.x)
            if exceeds_threshold:
                return None

        return instance

    def sample_drift_location(self):
        """
        Sample a random location for the drift onset within the specified region.
        
        Returns:
            float: The sampled location as a proportion of the stream length
        """
        loc = np.random.uniform(self.drift_region[0] + self.burn_in_samples,
                                self.drift_region[1] - self.burn_in_samples, 1)[0]

        return loc

    def _shuffle_arr(self, arr):
        """
        Shuffle the array using the pre-computed permutation indices.
        
        Args:
            arr (np.ndarray): The array to shuffle
            
        Returns:
            np.ndarray: The shuffled array
        """
        shuffled_arr = np.take(arr, self.fitted['x_perm_idx'])

        return shuffled_arr

    def _arr_exceeds_threshold(self, arr):
        """
        Check if any element in the array exceeds the specified threshold.
        
        Args:
            arr (np.ndarray): The array to clip
            
        Returns:
            bool: True if any element exceeds the threshold, False otherwise
        """
        exceed_idx = self.fitted['x_exceed_idx']
        exceed_val = self.fitted['x_exceed_val']

        if arr[exceed_idx] > exceed_val:
            return True

        return False

    def _skip_instance(self):
        """
        Determine whether to skip an instance based on the skip probability.
        
        Returns:
            bool: True if the instance should be skipped, False otherwise
        """
        return np.random.binomial(1, 1 - self.label_skip_proba) < 1

    @classmethod
    def select_random_num_attr(cls, schema):
        # numeric_attrs = schema.get_numeric_attributes()
        numeric_attrs = [*STREAM_MEDIANS[schema.dataset_name]]

        selected_attr = random.choice(numeric_attrs)

        selected_attr_idx = cls.get_attr_position(schema, selected_attr)

        return selected_attr, selected_attr_idx

    @staticmethod
    def get_attr_position(schema, attr_name):
        i = 0
        for i in range(schema.get_num_attributes()):
            if schema._moa_header.attribute(i).name() == attr_name:
                break

        return i

    @staticmethod
    def shuffle_stream(stream, max_n_instances: Optional[int] = None):
        """
        Create a new stream by randomly shuffling all instances.
        
        Args:
            stream: The input stream to shuffle
            
        Returns:
            NumpyStream: A new stream with shuffled instances
            :param max_n_instances: max n instances
        """

        sch = stream.get_schema()

        attr_names = [sch._moa_header.attribute(i).name()
                      for i in range(sch.get_num_attributes())]

        X_list, y_list = [], []
        instance_processed = 0
        while stream.has_more_instances():
            if max_n_instances is not None:
                if instance_processed > max_n_instances:
                    break

            instance = stream.next_instance()

            X_list.append(instance.x)
            y_list.append(instance.y_index)
            instance_processed += 1

        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)

        shuffle_idx = np.random.permutation(X.shape[0])
        X_shuffled = X.iloc[shuffle_idx]
        X_shuffled.columns = attr_names
        y_shuffled = y.iloc[shuffle_idx]

        np_stream = NumpyStream(X=X_shuffled.values, y=y_shuffled.values,
                                dataset_name=stream.get_schema().dataset_name,
                                feature_names=X_shuffled.columns)

        return np_stream
