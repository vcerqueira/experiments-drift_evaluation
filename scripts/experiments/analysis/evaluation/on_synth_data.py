import pandas as pd

from src.misc import prep_latex_tab, DataReader


MODE = 'ABRUPT'

df_f1 = DataReader.get_synth_results(metric='f1', round_to=3)
df_f1 = df_f1.drop(columns=['Classifier'])
df_f1 = df_f1.query(f'Mode=="{MODE}"').drop(columns=['Mode'])
# df_f1 = df_f1.set_index(['Stream', 'Mode'])
df_f1 = df_f1.set_index(['Stream'])

df_f1_tab = prep_latex_tab(df_f1,
                           rotate_cols=False,
                           minimize=False)

# df_er = DataReader.get_synth_results(metric='episode_recall', round_to=4)
# df_er_tab = prep_latex_tab(df_er.set_index(['Stream', 'Classifier']),
#                            rotate_cols=True,
#                            minimize=False)

df_far = DataReader.get_synth_results(metric='far', round_to=3)
df_far = df_far.drop(columns=['Classifier'])
df_far = df_far.query(f'Mode=="{MODE}"').drop(columns=['Mode'])
# df_far = df_far.set_index(['Stream', 'Mode'])
df_far = df_far.set_index(['Stream'])

df_far_tab = prep_latex_tab(df_far,
                            rotate_cols=False,
                            minimize=True)

combined_df = pd.concat({'F1': df_f1_tab, 'FAR': df_far_tab})
combined_df = combined_df.swaplevel(0, 1).sort_index()


latex_table = combined_df.T.to_latex(
    float_format="%.3f",
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    caption=f'Performance metrics of drift detectors across different datasets {MODE}',
    label='tab:combined_metrics',
    position='htbp'
)

print(latex_table)
