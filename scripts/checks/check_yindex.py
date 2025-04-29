import numpy as np
import pandas as pd
from capymoa.datasets import (ElectricityTiny,
                              CovtypeTiny,
                              Sensor)

from src.streams.inject_drift import DriftSimulator

# stream = ElectricityTiny()
stream = CovtypeTiny()
# stream = Sensor()

sch = stream.get_schema()
n = stream._length

print(sch)
print(sch.get_num_attributes())
np.random.choice(sch.get_num_attributes())
sch.get_num_numeric_attributes()
sch.get_numeric_attributes()

numeric_attrs = sch.get_numeric_attributes()


selected_attr, selected_attr_idx = DriftSimulator.select_random_num_attr(sch)

instance = stream.next_instance()

instance.x[selected_attr_idx]


instance.schema.get_moa_header()
instance.schema.__str__()



instance.x
