from typing import List, Union, Tuple, Dict

import numpy as np

ArrayLike = Union[List[int], List[Tuple[int, int]], np.ndarray[int]]


class EvaluateDetector:
    """A class to evaluate the performance of concept drift detectors.

    This class provides functionality to assess drift detection algorithms by comparing
    their predictions against ground truth drift points. Each drift point is represented
    as a tuple (start_location, end_location) to handle both abrupt and gradual drifts.

    Key Features:
        - Handles both abrupt and gradual drifts:
            * Abrupt drifts: start_location = end_location, e.g., (100, 100)
            * Gradual drifts: start_location < end_location, e.g., (100, 150)
        - Considers maximum acceptable detection delay
        - Calculates comprehensive performance metrics (precision, recall, F1)
        - Tracks false alarm rates and mean time to detection

    Attributes:
        max_delay (int): Maximum allowable delay for drift detection
        metrics (dict): Dictionary storing the latest calculated performance metrics
            - fp (int): False positive count
            - tp (int): True positive count
            - fn (int): False negative count
            - precision (float): Precision score
            - recall (float): Recall score
            - f1 (float): F1 score
            - mtd (float): Mean time to detect
            - fa_1k (float): False alarms per 1000 instances
            - n_episodes (int): Number of drift episodes
            - n_alarms (int): Total number of alarms raised

    Example:
        >>> evaluator = EvaluateDetector(max_delay=50)
        >>> metrics = evaluator.calc_performance(
        ...     trues=[(100, 100),   # Abrupt drift at position 100
        ...            (200, 250)],   # Gradual drift from position 200 to 250
        ...     preds=[98, 205],      # Predicted drift points
        ...     tot_n_instances=1000
        ... )
        >>> print(f"F1 Score: {metrics['f1']:.2f}")
    """

    def __init__(self, max_delay: int):
        """Initialize the drift detector evaluator.

        Args:
            max_delay (int): Maximum number of instances to wait for a detection after a drift
                occurs. For gradual drifts, this window starts from the drift end_location.
                If a detector fails to signal within this window, it is considered to have
                missed the drift (false negative).

        Raises:
            ValueError: If max_delay is not a positive integer.

        Note:
            - The max_delay parameter is crucial for evaluating both the accuracy and speed
              of drift detection.
            - For gradual drifts (where start_location != end_location), the detection
              window extends from (start_location - max_delay) to (end_location + max_delay).
            - For abrupt drifts (where start_location = end_location), the detection
              window is (drift_point - max_delay) to (drift_point + max_delay).
        """
        if not isinstance(max_delay, int) or max_delay <= 0:
            raise ValueError('max_delay must be a positive integer')

        self.max_delay = max_delay
        self.metrics = {}

    def calc_performance(self, trues: ArrayLike, preds: ArrayLike, tot_n_instances: int) -> Dict:
        """Calculate performance metrics for drift detection.

        Evaluates drift detection performance by comparing predicted drift points against
        true drift points, considering a maximum allowable delay. Calculates various metrics
        including precision, recall, F1-score, mean time to detect (MTD), and false alarm rate.

        todo count consecutive false alarms?

        Args:
            trues: Array-like of true drift points represented as (start, end) tuples
                indicating drift intervals. For gradual drifts, the end point can be
                different from the start point.
            preds: Array-like of predicted drift points (indices) where the detector
                signaled a drift.
            tot_n_instances: Total number of instances in the data stream, used to
                calculate false alarm rate.

        Returns:
            Dict containing the following metrics:
                - fp (int): False positives (incorrect detections)
                - tp (int): True positives (correct detections)
                - fn (int): False negatives (missed drifts)
                - precision (float): Precision score (tp / (tp + fp))
                - recall (float): Recall score (tp / (tp + fn))
                - f1 (float): F1 score (harmonic mean of precision and recall)
                - mtd (float): Mean time to detect successful detections
                - fa_1k (float): False alarms per 1000 instances
                - n_episodes (int): Total number of drift episodes
                - n_alarms (int): Total number of alarms raised

        Raises:
            ValueError: If arrays are not ordered or contain invalid values
            AssertionError: If no drift points are provided
        """

        self._check_arrays(trues, preds)
        if tot_n_instances <= 0:
            raise ValueError('Total number of instances must be positive')

        fp, tp, fn = 0, 0, 0
        efp, etp, efn = 0, 0, 0
        detection_times: List[float] = []
        n_episodes, n_alarms = 0, 0

        drift_eps = self._get_drift_episodes(trues=trues, preds=preds)

        for episode in drift_eps:
            n_episodes += 1
            drift_detected = False
            episode_detection_time = np.nan
            drift_start, drift_end = episode['true']

            for pred in episode['preds']:
                # print(pred)
                n_alarms += 1

                if drift_start - self.max_delay <= pred <= drift_end + self.max_delay:
                    tp += 1
                    if not drift_detected:  # only counting first detection
                        drift_detected = True
                        episode_detection_time = pred - drift_start
                else:
                    fp += 1

            if drift_detected:
                etp += 1
                detection_times.append(episode_detection_time)
            else:
                fn += 1

        precision, recall, f1 = self._calc_classification_metrics(tp=tp, fp=fp, fn=fn)
        false_alarm_rate = (fp / tot_n_instances) * 1000
        alarm_rate = (n_alarms / tot_n_instances) * 1000
        mean_detection_time = np.nanmean(detection_times) if detection_times else np.nan
        ep_recall = etp / n_episodes

        self.metrics = {
            'fp': fp,
            'tp': tp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'ep_recall': ep_recall,
            'f1': f1,
            'mdt': mean_detection_time,
            'fa_1k': false_alarm_rate,
            'alarms_per_1k': alarm_rate,
            'n_episodes': n_episodes,
            'n_alarms': n_alarms,
        }

        return self.metrics

    def _get_drift_episodes(self, trues: List, preds: ArrayLike) -> List[Dict]:
        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        if not isinstance(trues, np.ndarray):
            trues = np.asarray(trues)

        next_starting_point = 0
        drift_episodes = []
        for true in trues:
            episode_preds = preds[preds > next_starting_point]
            drift_start, drift_end = true

            episode_preds = episode_preds[episode_preds <= drift_end + self.max_delay]
            episode_preds -= next_starting_point

            drift_episodes.append(
                {'preds': episode_preds,
                 'true': (drift_start - next_starting_point,
                          drift_end - next_starting_point)}
            )

            next_starting_point = drift_end + self.max_delay

        return drift_episodes

    @staticmethod
    def _check_arrays(trues: ArrayLike, preds: ArrayLike):
        assert len(trues) > 0, 'No drift points given'

        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        if not isinstance(trues, np.ndarray):
            trues = np.asarray(trues)

        if len(preds) > 1:
            tot_neg_alarms = np.sum(np.diff(preds) < 0)
            if tot_neg_alarms > 0:
                raise ValueError('Provide an ordered list of detections')

        if len(trues) > 1:
            tot_neg_drifts = np.sum(np.diff(trues) < 0)
            if tot_neg_drifts > 0:
                raise ValueError('Provide an ordered list of drift points')

    @staticmethod
    def _calc_classification_metrics(tp, fp, fn):

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1_score
