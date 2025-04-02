"""
Incremental Drift Simulator

This module provides functionality to simulate incremental concept drift
in datasets by sequentially sampling and rearranging data based on a drifting feature.
"""

import random
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Any
from scipy.stats import pearsonr, spearmanr

from capymoa.instance import LabeledInstance
from capymoa.stream import NumpyStream


def check_feature_correlation(data: pd.DataFrame, 
                             target_index: int = -1, 
                             method: str = 'pearson') -> str:
    """
    Find the feature with the highest correlation to the target variable.
    
    Args:
        data: DataFrame containing the dataset
        target_index: Index of the target column (-1 for last column)
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Name of the feature with highest correlation to the target
    """
    # Extract target column
    if target_index < 0:
        target_index = data.shape[1] + target_index
    
    target = data.iloc[:, target_index]
    features = data.drop(data.columns[target_index], axis=1)
    
    # Only consider numeric features
    numeric_features = features.select_dtypes(include=['number']).columns
    
    if len(numeric_features) == 0:
        raise ValueError("No numeric features found in the dataset")
    
    correlations = {}
    for feature in numeric_features:
        if method == 'pearson':
            corr, _ = pearsonr(features[feature], target)
        elif method == 'spearman':
            corr, _ = spearmanr(features[feature], target)
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        correlations[feature] = abs(corr)
    
    # Return the feature with the highest absolute correlation
    return max(correlations, key=correlations.get)


def fold_dataframe(df: pd.DataFrame, 
                  column: str, 
                  step: float, 
                  frac: float = 0.5, 
                  higher: str = 'left', 
                  shape: str = '^') -> List[pd.DataFrame]:
    """
    Fold a DataFrame into two parts based on a drifting feature.
    
    Args:
        df: DataFrame to fold
        column: Column to use for folding
        step: Threshold value for splitting
        frac: Fraction for splitting (default 0.5)
        higher: Which side should have higher values ('left' or 'right')
        shape: Shape of the fold ('^' for ascending, 'v' for descending)
        
    Returns:
        List of folded DataFrames
    """
    # Sort by the column
    df_sorted = df.sort_values(by=column).reset_index(drop=True)
    
    # Split point
    split_idx = int(len(df) * frac)
    
    # Split the dataframe
    left = df_sorted.iloc[:split_idx].copy()
    right = df_sorted.iloc[split_idx:].copy()
    
    # Adjust based on the 'higher' parameter
    if higher == 'right':
        left, right = right, left
    
    # Adjust based on shape
    if shape == 'v':
        left = left.iloc[::-1].reset_index(drop=True)
    elif shape == '^':
        right = right.iloc[::-1].reset_index(drop=True)
    
    return [left, right]


def fold_multiple_times(dfs: List[pd.DataFrame], 
                       fold_function, 
                       times: int, 
                       column: str) -> List[pd.DataFrame]:
    """
    Repeatedly fold DataFrames to create multiple concepts.
    
    Args:
        dfs: List of DataFrames to fold
        fold_function: Function used for folding
        times: Number of additional folds to create
        column: Column used for folding
        
    Returns:
        List of folded DataFrames
    """
    if times <= 0:
        return dfs
    
    result = []
    for df in dfs:
        if len(df) > 100:  # Only fold if there's enough data
            step = df[column].quantile(0.5)
            frac = random.uniform(0.4, 0.6)
            higher = 'left' if random.random() > 0.5 else 'right'
            shape = '^' if random.random() > 0.5 else 'v'
            
            folded = fold_function(df, column, step, frac, higher, shape)
            result.extend(folded)
        else:
            result.append(df)
    
    return fold_multiple_times(result, fold_function, times - 1, column)


def extend_indices_to_spans(indices: List[int], initial_length: int) -> List[int]:
    """
    Convert a list of indices to spans with specified initial length.
    
    Args:
        indices: List of indices
        initial_length: Length of the first span
        
    Returns:
        List of span boundaries
    """
    if not indices:
        return []
    
    spans = [indices[0]]
    current_length = initial_length
    
    for i in range(1, len(indices)):
        span_end = spans[-1] + current_length
        spans.append(min(span_end, indices[i]))
        current_length = indices[i] - spans[-1]
    
    spans.append(indices[-1] + current_length)
    return spans


class IncrementalDriftSimulator:
    """
    Simulator for incremental concept drift by rearranging data based on a drifting feature.
    """
    
    def __init__(self, 
                schema,
                num_drifts: int = 2,
                drift_lengths: Union[int, List[int]] = 1000,
                drifting_feature: Optional[str] = None,
                target_index: int = -1,
                correlation_method: str = 'pearson',
                drop_drifting_feature: bool = True):
        """
        Initialize the incremental drift simulator.
        
        Args:
            schema: Stream schema
            num_drifts: Number of drift points to simulate
            drift_lengths: Length of each drift period
            drifting_feature: Feature to use for drift simulation (if None, one will be selected)
            target_index: Index of the target variable
            correlation_method: Method for selecting drifting feature ('pearson' or 'spearman')
            drop_drifting_feature: Whether to drop the drifting feature from output
        """
        self.schema = schema
        self.num_drifts = num_drifts
        self.drift_lengths = drift_lengths
        self.drifting_feature = drifting_feature
        self.target_index = target_index
        self.correlation_method = correlation_method
        self.drop_drifting_feature = drop_drifting_feature
        
        self.fitted = False
        self.drifting_feature_idx = None
        self.drift_points = []
        self.data_buffer = []
        self.data_chunks = []
        self.feature_names = None
        self.instances_processed = 0
        
    def collect_data(self, instance: LabeledInstance) -> None:
        """
        Collect data instances for later processing.
        
        Args:
            instance: Data instance to collect
        """
        if not self.fitted:
            self.data_buffer.append((instance.x, instance.y_index))
    
    def fit(self) -> None:
        """
        Fit the simulator by processing collected instances and setting up drift points.
        """
        if len(self.data_buffer) == 0:
            raise ValueError("No data collected. Call collect_data() first.")
        
        # Convert buffer to dataframe
        X = np.array([x for x, _ in self.data_buffer])
        y = np.array([y for _, y in self.data_buffer])
        
        # Get feature names from schema
        self.feature_names = [f"attr_{i}" for i in range(X.shape[1])]
        
        # Create dataframe
        df = pd.DataFrame(X, columns=self.feature_names)
        df['target'] = y
        
        # Determine drifting feature if not specified
        if self.drifting_feature is None:
            self.drifting_feature = check_feature_correlation(
                df, 
                target_index=-1, 
                method=self.correlation_method
            )
            print(f"Selected drifting feature: {self.drifting_feature}")
        
        self.drifting_feature_idx = self.feature_names.index(self.drifting_feature)
        
        # Process drift lengths
        if isinstance(self.drift_lengths, int):
            self.drift_lengths = [self.drift_lengths] * self.num_drifts
        elif len(self.drift_lengths) != self.num_drifts:
            raise ValueError("Length of drift_lengths list must match the number of drifts.")
        
        # Sort data by drifting feature
        df = df.sort_values(by=self.drifting_feature).reset_index(drop=True)
        
        # Calculate drift points
        total_drift_length = sum(self.drift_lengths)
        remaining_length = len(df) - total_drift_length
        
        if remaining_length <= 0:
            raise ValueError("Total drift length exceeds dataset size.")
        
        # Calculate lengths of non-drifting parts
        num_non_drifting_parts = self.num_drifts + 1
        non_drifting_lengths = [remaining_length // num_non_drifting_parts] * num_non_drifting_parts
        
        # Adjust last part to account for rounding
        non_drifting_lengths[-1] += remaining_length - sum(non_drifting_lengths)
        
        # Calculate drift points
        current_pos = 0
        for i in range(self.num_drifts):
            stable_end = current_pos + non_drifting_lengths[i]
            drift_start = stable_end
            drift_end = drift_start + self.drift_lengths[i]
            
            self.drift_points.append((drift_start, drift_end))
            current_pos = drift_end
        
        # Create data chunks
        step = df[self.drifting_feature].quantile(0.2)
        
        fold_times = 0
        if self.num_drifts > 1:
            decider = random.random()
            if decider > 0.5:
                higher = 'left'
                frac = 0.55
            else:
                higher = 'right'
                frac = 0.45
            
            shape = '^' if random.random() > 0.5 else 'v'
            folded_df = fold_dataframe(df, self.drifting_feature, step, frac, higher, shape)
        else:
            folded_df = [df]
        
        fold_times += 1
        
        # Fold multiple times if needed
        lst_concept = fold_multiple_times(
            folded_df, 
            fold_dataframe, 
            times=self.num_drifts - 2, 
            column=self.drifting_feature
        )
        
        # Calculate local peaks for chunks
        local_peaks = [0]
        for chunk in lst_concept:
            local_peaks.append(local_peaks[-1] + len(chunk))
        
        # Create spans for each concept
        list_of_spans = extend_indices_to_spans(local_peaks, non_drifting_lengths[0])
        
        # Merge all chunks
        merged_df = pd.concat(lst_concept, ignore_index=True)
        
        # Create final chunks
        combined = []
        for i in range(len(list_of_spans) - 1):
            chunk = merged_df.iloc[list_of_spans[i]:list_of_spans[i + 1]]
            
            # Shuffle stable concepts
            if i % 2 == 0:
                chunk = chunk.sample(frac=1).reset_index(drop=True)
            
            combined.append(chunk)
        
        # Final dataset
        result = pd.concat(combined, ignore_index=True)
        
        if self.drop_drifting_feature:
            result = result.drop(columns=[self.drifting_feature])
            # Adjust feature index list
            self.feature_names.pop(self.drifting_feature_idx)
        
        # Split back into X and y
        y_result = result['target'].values
        X_result = result.drop(columns=['target']).values
        
        # Store as data chunks
        for i in range(len(X_result)):
            self.data_chunks.append((X_result[i], y_result[i]))
        
        self.fitted = True
        self.instances_processed = 0
    
    def transform(self, instance: LabeledInstance) -> LabeledInstance:
        """
        Transform an instance according to the simulated drift pattern.
        
        Args:
            instance: Instance to transform
            
        Returns:
            Transformed instance
        """
        if not self.fitted:
            self.collect_data(instance)
            return instance
        
        if self.instances_processed >= len(self.data_chunks):
            # If we've used all the data, just return the original instance
            return instance
        
        # Get the appropriate chunk
        X, y = self.data_chunks[self.instances_processed]
        self.instances_processed += 1
        
        # Create a new instance
        if self.drop_drifting_feature:
            # If drifting feature was dropped, we need to recreate X without it
            original_x = instance.x
            new_x = np.delete(original_x, self.drifting_feature_idx)
            new_x = X  # Replace with the transformed data
        else:
            new_x = X
        
        # Return transformed instance
        return LabeledInstance.from_array(self.schema, new_x, y)
    
    def get_drift_points(self) -> List[Tuple[int, int]]:
        """
        Get the drift points as (start, end) tuples.
        
        Returns:
            List of drift points as (start, end) tuples
        """
        return self.drift_points
    
    def reset(self) -> None:
        """
        Reset the simulator to its initial state.
        """
        self.instances_processed = 0
    
    @classmethod
    def simulate_incremental_drift(cls,
                                   concept: Union[pd.DataFrame, str],
                                   num_drifts: int = 2,
                                   drift_lengths: Union[int, List[int]] = 1000,
                                   drifting_feature: Optional[str] = None,
                                   target_index: int = -1,
                                   method: str = 'pearson',
                                   drop_drifting_feature: bool = True) -> Tuple[pd.DataFrame, str]:
        """
        Simulate incremental drifts in a dataset by sequentially sampling.
        
        Args:
            concept: DataFrame or path to dataset
            num_drifts: Number of drifts to simulate
            drift_lengths: Length of each drift period
            drifting_feature: Feature to use for drift simulation
            target_index: Index of target column
            method: Correlation method for feature selection
            drop_drifting_feature: Whether to drop the drifting feature
            
        Returns:
            Tuple of (DataFrame with simulated drift, name of drifting feature)
        """
        # Load data if file path is given
        if isinstance(concept, str):
            if concept.endswith('.csv'):
                concept = pd.read_csv(concept)
            elif concept.endswith('.arff'):
                from scipy.io import arff
                data, meta = arff.loadarff(concept)
                concept = pd.DataFrame(data)
        
        # Determine drifting feature if None
        if drifting_feature is None:
            drifting_feature = check_feature_correlation(concept, target_index, method)
        
        if drifting_feature not in concept.columns:
            raise ValueError(f"The feature '{drifting_feature}' is not present in the concept.")
        
        # Handle drift lengths
        if isinstance(drift_lengths, int):
            drift_lengths = [drift_lengths] * num_drifts
        elif isinstance(drift_lengths, list) and len(drift_lengths) != num_drifts:
            raise ValueError("Length of drift_lengths list must match the number of drifts.")
        
        # Sort the entire dataset by the drifting feature
        concept = concept.sort_values(by=drifting_feature).reset_index(drop=True)
        
        total_drift_length = sum(drift_lengths)
        remaining_length = len(concept) - total_drift_length
        if remaining_length <= 0:
            raise ValueError("Total drift length exceeds dataset size.")
        
        # Calculate lengths of non-drifting parts
        num_non_drifting_parts = num_drifts + 1
        non_drifting_lengths = [remaining_length // num_non_drifting_parts] * num_non_drifting_parts
        list_of_parts_length = []
        for i in range(num_drifts):
            list_of_parts_length.append(non_drifting_lengths[i])
            list_of_parts_length.append(drift_lengths[i])
        list_of_parts_length.append(len(concept) - sum(list_of_parts_length))
        
        if sum(list_of_parts_length) != len(concept):
            raise ValueError("The sum of the lengths of the parts does not match the length of the concept.")
        
        # Split the concept into parts
        drifting_feature_values = concept[drifting_feature]
        step = drifting_feature_values.quantile(0.2)
        
        fold_times = 0
        if num_drifts > 1:
            decider = random.random()
            if decider > 0.5:
                higher = 'left'
                frac = 0.55
            else:
                higher = 'right'
                frac = 0.45
            
            shape = '^' if random.random() > 0.5 else 'v'
            folded_concept = fold_dataframe(concept, drifting_feature, step, frac, higher, shape)
        else:
            folded_concept = [concept]
        
        fold_times += 1
        
        lst_concept = fold_multiple_times(folded_concept, fold_dataframe, times=num_drifts - 2, column=drifting_feature)
        local_peaks = [0]
        for l in lst_concept:
            local_peaks.append(local_peaks[-1] + len(l))
        
        list_of_spans = extend_indices_to_spans(local_peaks, non_drifting_lengths[0])
        
        merged_df = pd.concat(lst_concept, ignore_index=True)
        combined = []
        for i in range(len(list_of_spans) - 1):
            chunk = merged_df.iloc[list_of_spans[i]:list_of_spans[i + 1]]
            if i % 2 == 0:
                chunk = chunk.sample(frac=1).reset_index(drop=True)
            combined.append(chunk)
        
        result = pd.concat(combined, ignore_index=True)
        if drop_drifting_feature:
            result = result.drop(columns=[drifting_feature])
        
        return result, drifting_feature
    
    
class IncrementalDriftStream:
    """
    Stream wrapper for incremental drift simulation on any base stream.
    """
    
    def __init__(self,
                base_stream,
                num_drifts: int = 2,
                drift_lengths: Union[int, List[int]] = 1000,
                drifting_feature: Optional[str] = None,
                correlation_method: str = 'pearson',
                drop_drifting_feature: bool = True):
        """
        Initialize an incremental drift stream.
        
        Args:
            base_stream: The base stream to apply drift to
            num_drifts: Number of drifts to simulate
            drift_lengths: Length of each drift period
            drifting_feature: Feature to use for drift (if None, one will be selected)
            correlation_method: Method for selecting drifting feature
            drop_drifting_feature: Whether to drop the drifting feature
        """
        self.base_stream = base_stream
        self.schema = base_stream.get_schema()
        
        self.simulator = IncrementalDriftSimulator(
            schema=self.schema,
            num_drifts=num_drifts,
            drift_lengths=drift_lengths,
            drifting_feature=drifting_feature,
            correlation_method=correlation_method,
            drop_drifting_feature=drop_drifting_feature
        )
        
        self.collection_phase = True
        self.transform_phase = False
        self.collected_instances = []
        
    def prepare(self, max_instances: int = 10000) -> None:
        """
        Prepare the stream by collecting instances and fitting the simulator.
        
        Args:
            max_instances: Maximum number of instances to collect
        """
        if not self.collection_phase:
            return
        
        # Collect instances for fitting
        instances_collected = 0
        while self.base_stream.has_more_instances() and instances_collected < max_instances:
            instance = self.base_stream.next_instance()
            self.simulator.collect_data(instance)
            self.collected_instances.append(instance)
            instances_collected += 1
        
        # Fit the simulator
        self.simulator.fit()
        
        # Reset the stream state
        self.collection_phase = False
        self.transform_phase = True
        self.base_stream.restart()
    
    def next_instance(self) -> LabeledInstance:
        """
        Get the next instance from the stream.
        
        Returns:
            Transformed instance with simulated drift
        """
        if self.collection_phase:
            self.prepare()
        
        if self.base_stream.has_more_instances():
            instance = self.base_stream.next_instance()
            return self.simulator.transform(instance)
        
        return None
    
    def has_more_instances(self) -> bool:
        """
        Check if the stream has more instances.
        
        Returns:
            True if more instances are available
        """
        return self.base_stream.has_more_instances() or self.collection_phase
    
    def restart(self) -> None:
        """
        Restart the stream.
        """
        self.base_stream.restart()
        self.simulator.reset()
    
    def get_schema(self):
        """
        Get the stream schema.
        
        Returns:
            Stream schema
        """
        return self.schema
    
    def get_drifts(self):
        """
        Get the drift points in the stream.
        
        Returns:
            List of drift points
        """
        if self.collection_phase:
            self.prepare()
        
        return self.simulator.get_drift_points() 