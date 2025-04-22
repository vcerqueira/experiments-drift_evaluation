import pandas as pd


def read_real(dataset: str, type: str, learner: str, metric: str):
    df = pd.read_csv(f'{dataset},{type},{learner}.csv').set_index('Unnamed: 0')
    df.index.name = 'Detector'

    df_metric = df[metric]

    return df_metric


def read_real2(dataset: str, type: str, learner: str, metric: str):
    df = pd.read_csv(f'assets/results/real/{dataset},{type},{learner}.csv').set_index('Unnamed: 0')
    df.index.name = 'Detector'

    df_metric = df[metric]

    return df_metric


df_opt = read_real('Electricity', 'x_exceed_skip', 'HoeffdingTree', 'f1')
# df_opt = read_real('Electricity', 'x_permutations', 'HoeffdingTree', 'f1')
df_van = read_real2('Electricity', 'x_exceed_skip', 'HoeffdingTree', 'f1')
# df_van = read_real2('Electricity', 'x_permutations', 'HoeffdingTree', 'f1')
print(df_opt)
print(df_van)

df_opt - df_van
