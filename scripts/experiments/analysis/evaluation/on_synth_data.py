import pandas as pd

from utils.misc import prep_latex_tab, DataReader

df_f1 = DataReader.get_synth_results(metric='f1', round_to=4)
df_f1_tab = prep_latex_tab(df_f1.set_index(['Stream', 'Classifier']),
                           rotate_cols=True,
                           minimize=False)

df_er = DataReader.get_synth_results(metric='episode_recall', round_to=4)
df_er_tab = prep_latex_tab(df_er.set_index(['Stream', 'Classifier']),
                           rotate_cols=True,
                           minimize=False)

df_far = DataReader.get_synth_results(metric='far', round_to=4)
df_far_tab = prep_latex_tab(df_far.set_index(['Stream', 'Classifier']),
                            rotate_cols=True,
                            minimize=True)

combined_df = pd.concat({'F1': df_f1_tab,
                         'ER': df_er_tab,
                         'FAR': df_far_tab})

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
