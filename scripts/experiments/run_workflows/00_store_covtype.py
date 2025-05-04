from pathlib import Path

import pandas as pd
from src.streams.real import CAPYMOA_DATASETS

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / 'data'
dataset_name = 'Covtype'

stream = CAPYMOA_DATASETS[dataset_name]()
sch = stream.get_schema()
attr_names = [str(sch._moa_header.attribute(i).name()) for i in range(sch.get_num_attributes())]

X_list, y_list = [], []
instance_processed = 0
while stream.has_more_instances():
    instance = stream.next_instance()

    X_list.append(instance.x)
    y_list.append(instance.y_index)
    instance_processed += 1

X = pd.DataFrame(X_list).copy()
y = pd.Series(y_list).copy()
X.columns = attr_names
X['target'] = y

file_path = OUTPUT_DIR / f'{dataset_name}-df.csv'

X.to_csv(file_path, index=False)
