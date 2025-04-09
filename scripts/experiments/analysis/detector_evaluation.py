import pandas as pd


def read_real(dataset: str, type: str, learner: str, metric: str):
    df = pd.read_csv(f'{dataset},{type},{learner}.csv').set_index('Unnamed: 0')
    df.index.name = 'Detector'

    df_metric = df[metric]

    return df_metric


df_y = read_real('Electricity', 'x_permutations', 'HoeffdingTree', 'f1')
df_y = read_real('Electricity', 'x_permutations', 'HoeffdingTree', 'far')
df_y = read_real('Electricity', 'x_permutations', 'ARF', 'f1')
df_y = read_real('Electricity', 'x_exceed_skip', 'HoeffdingTree', 'f1')
df_y = read_real('Electricity', 'x_exceed_skip', 'ARF', 'f1')
df_y = read_real('Electricity', 'x_exceed_skip', 'NaiveBayes', 'f1')
df_y = read_real('Electricity', 'y_prior_skip', 'HoeffdingTree', 'f1')
df_y = read_real('Electricity', 'y_prior_skip', 'NaiveBayes', 'f1')
df_y = read_real('Electricity', 'y_swaps', 'HoeffdingTree', 'f1')
df_y = read_real('Electricity', 'y_swaps', 'ARF', 'f1')
df_y = read_real('Electricity', 'y_swaps', 'NaiveBayes', 'f1')
print(df_y)

df_x = read_real('Electricity', 'ABRUPT@X', 'HoeffdingTree', 'f1')
df_xy = read_real('Electricity', 'ABRUPT@X', 'HoeffdingTree', 'f1')
df_1 = pd.concat([df_y, df_x, df_xy], axis=1).round(4)
df_1.columns = pd.MultiIndex.from_product([['Electricity'], ['Y', 'X', 'XY']])

df_y = read_real('Covtype', 'ABRUPT@Y', 'HoeffdingTree', 'f1')
df_x = read_real('Covtype', 'ABRUPT@X', 'HoeffdingTree', 'f1')
df_xy = read_real('Covtype', 'ABRUPT@X', 'HoeffdingTree', 'f1')
df_2 = pd.concat([df_y, df_x, df_xy], axis=1).round(4)
df_2.columns = pd.MultiIndex.from_product([['Covtype'], ['Y', 'X', 'XY']])


df = pd.concat([df_1,df_2],axis=1)


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


df = to_latex_tab(df.T, minimize=False).T


# Generate LaTeX table
latex_table = df.to_latex(
    float_format="%.3f",
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    caption='Performance metrics of drift detectors across different datasets',
    label='tab:combined_metrics',
    position='htbp'
)

print(latex_table)
