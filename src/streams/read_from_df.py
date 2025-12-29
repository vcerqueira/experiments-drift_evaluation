from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from capymoa.stream import NumpyStream


class StreamFromDF:
    # StreamFromDF.read_csv('data/Asfault.csv')

    @classmethod
    def read_stream(cls, stream_name: str, **kwargs):
        fp = f'data/{stream_name}.csv'

        return cls.read_csv(filepath=fp, **kwargs)

    @staticmethod
    def get_stream_name(filepath: str):
        return Path(filepath).stem

    @classmethod
    def read_csv(cls,
                 filepath: str,
                 max_n_instances: Optional[int] = 100_000,
                 shuffle: bool = True,
                 as_np_stream: bool = True):

        # filepath = 'data/Covtype.csv'

        dataset_name = cls.get_stream_name(filepath)

        if dataset_name in ['Covtype', 'Electricity']:
            header_ = 'infer'
        else:
            header_ = None

        # filepath = 'data/Asfault.csv'

        df = pd.read_csv(filepath, header=header_, on_bad_lines='skip')

        if dataset_name not in ['Covtype', 'Electricity']:
            df.columns = [f'F{i}' for i in range(1, df.shape[1])] + ['target']

        if shuffle:
            df = df.sample(n=df.shape[0], replace=False).reset_index(drop=True)

        if max_n_instances is not None:
            df = df.head(max_n_instances)

        if as_np_stream:
            X = df.drop(columns='target')
            y = df['target']
            y = LabelEncoder().fit_transform(y)

            np_stream = NumpyStream(X=X.values,
                                    y=y,
                                    dataset_name=dataset_name,
                                    feature_names=X.columns)

            return np_stream
        else:
            return df
