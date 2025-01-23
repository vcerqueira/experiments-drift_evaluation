import pandas as pd

perf = pd.read_csv('assets/results/detector_hypertuning,ABRUPT2.csv')
perf = perf.drop(columns=['Unnamed: 0'])

perf.groupby(['detector','stream']).apply(lambda x: x.sort_values('f1').iloc[-1,:]['f1'])
perf.groupby(['detector','stream']).apply(lambda x: x.sort_values('f1').iloc[-1,:])

# improve config space of EWMAChart, GeometricMovingAverage



