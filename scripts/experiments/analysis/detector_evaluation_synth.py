import pandas as pd


def read_synth(metric: str, round_to=None):
    STREAMS = ['Agrawal', 'STAGGER', 'SEA']
    results = {}
    for stream in STREAMS:
        df_ = pd.read_csv(f'assets/results/{stream},ABRUPT,ARF,POINT.csv').set_index('Unnamed: 0')
        # print(df_.columns.tolist())
        df_.index.name = 'Detector'

        results[stream] = df_[metric]

    df = pd.DataFrame(results)

    if round_to is not None:
        df = df.round(round_to)

    return df


def to_latex_tab(df, minimize: bool = False, rotate_cols: bool = False):
    if rotate_cols:
        df.columns = [f'\\rotatebox{{90}}{{{x}}}' for x in df.columns]

    annotated_res = []
    for i, r in df.iterrows():
        top_2 = r.sort_values(ascending=minimize).unique()[:2]
        if len(top_2) < 2:
            raise ValueError('only one score')

        best1 = r[r == top_2[0]].values[0]
        best2 = r[r == top_2[1]].values[0]

        r[r == top_2[0]] = f'\\textbf{{{best1}}}'
        r[r == top_2[1]] = f'\\underline{{{best2}}}'

        annotated_res.append(r)

    annotated_res = pd.DataFrame(annotated_res).astype(str)

    # text_tab = annotated_res.to_latex(caption='CAPTION', label='tab:scores_by_ds')

    return annotated_res


df_f1 = read_synth('f1', round_to=4)
df_f1 = to_latex_tab(df_f1.T, minimize=False).T
df_er = read_synth('ep_recall', round_to=4)
df_er = to_latex_tab(df_er.T, minimize=False).T
df_far = read_synth('fa_1k', round_to=4)
df_far = to_latex_tab(df_far.T, minimize=True).T

# print(df.mean(axis=1).sort_values())
# df.rank(axis=1).mean()
# df.rank(ascending=False).mean(axis=1).sort_values()


combined_df = pd.DataFrame(index=df_f1.index)

for dataset in ['Agrawal', 'STAGGER', 'SEA']:
    combined_df[(dataset, 'F1')] = df_f1[dataset]
    combined_df[(dataset, 'Recall')] = df_er[dataset]
    combined_df[(dataset, 'FAR')] = df_far[dataset]

# Create proper MultiIndex for columns
combined_df.columns = pd.MultiIndex.from_product([['Agrawal', 'STAGGER', 'SEA'], ['F1', 'Recall', 'FAR']])
combined_df.index.name = 'Detector'

# Generate LaTeX table
latex_table = combined_df.to_latex(
    float_format="%.3f",
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    caption='Performance metrics of drift detectors across different datasets',
    label='tab:combined_metrics',
    position='htbp'
)

print(latex_table)
