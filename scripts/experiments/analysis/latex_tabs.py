import pandas as pd

from src.streams.config import MAX_DELAY, DRIFT_WIDTH
from src.config import DETECTOR_PARAM_SPACE

N_SAMPLES = {'Asfault': 8563,
             'Covtype': 100000,
             'Electricity': 45312,
             'GasSensorArray': 13910,
             'NOAA': 18159,
             'Posture': 100000,
             'Rialto': 82250}

dataset_df = pd.DataFrame([N_SAMPLES, MAX_DELAY, DRIFT_WIDTH]).T.sort_index()
dataset_df.columns = ['# Samples', 'Max Delay', 'Drift Width']

dataset_df_tab = dataset_df.to_latex(caption='Dataset size and drift parameters', label='tab:data_specs')
print(dataset_df_tab)

rows = []
for detector, params in DETECTOR_PARAM_SPACE.items():
    for param_name, values in params.items():
        clean_param = param_name.replace('_', r'\_')

        val_str = ', '.join(map(str, values))
        rows.append({
            'Detector': detector,
            'Parameter': clean_param,
            'Values': val_str
        })

df = pd.DataFrame(rows)
df.set_index(['Detector', 'Parameter'], inplace=True)

latex_output = df.to_latex(
    index=True,
    column_format='llp{8cm}',
    caption='Detector Parameter Search Space',
    label='tab:detector_params',
    multirow=True,
)

print(latex_output)
