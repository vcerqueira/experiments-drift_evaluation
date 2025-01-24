from ast import literal_eval
from pprint import pprint

import pandas as pd

perf = pd.read_csv('assets/results/detector_hypertuning,ABRUPT.csv')
perf = perf.query('learner=="ARF"')
# perf.groupby(['detector', 'stream']).apply(lambda x: x.sort_values('f1').iloc[-1, :]['f1'])
# perf.groupby(['detector', 'stream']).apply(lambda x: x.sort_values('f1').iloc[-1, :])


stream_list = perf['stream'].unique().tolist()
detector_list = perf['detector'].unique().tolist()

loo_best_configs = {}
for stream in stream_list:
    # stream
    perf_loo = perf.query(f'stream!="{stream}"')

    detectors_f1 = {}
    for detector in detector_list:
        perf_dct = perf_loo.query(f'detector=="{detector}"')

        params_f1 = perf_dct.groupby('params').mean(numeric_only=True)['f1']

        best_config = literal_eval(params_f1.sort_values().index[-1])

        detectors_f1[detector] = best_config

    loo_best_configs[stream] = detectors_f1

pprint(loo_best_configs)
