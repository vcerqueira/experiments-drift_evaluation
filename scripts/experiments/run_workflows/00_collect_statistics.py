from pprint import pprint

import numpy as np
import pandas as pd
from capymoa.datasets import (ElectricityTiny, Electricity,
                              CovtypeTiny, CovtypeNorm,
                              Sensor)

from utils.streams.inject_drift import DriftSimulator

streams = {
    'ElectricityTiny': Electricity(),
    'CovtypeTiny': CovtypeNorm(),
    # 'Sensor': Sensor()
}

stream_stats = {}
for stream_name, stream in streams.items():
    print(stream_name)
    sch = stream.get_schema()
    print(sch.dataset_name)

    numeric_attrs = sch.get_numeric_attributes()

    attr_positions = {attr: DriftSimulator.get_attr_position(sch, attr) for attr in numeric_attrs}

    X_list = []
    while stream.has_more_instances():
        instance = stream.next_instance()

        X_list.append(instance.x)

    X = pd.DataFrame(X_list)

    median_values = X.median().values

    median_values_s = {attr: median_values[attr_positions[attr]] for attr in attr_positions}

    stream_stats[stream_name] = median_values_s

pprint(stream_stats)
