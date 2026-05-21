import pandas as pd

from src.misc import prep_latex_tab, DataReader

# MODE = 'GRADUAL'

df_f1 = DataReader.get_synth_results(metric='f1',
                                     round_to=3,
                                     stream_list=['Agrawal', 'SEA', 'STAGGER'],
                                     learners=['HoeffdingTree'])
df_f1 = df_f1.drop(columns=['Classifier'])
df_f1 = df_f1.set_index(['Mode', 'Stream']).sort_index().T#.astype(str)

# df_f1_tab = prep_latex_tab(df_f1,
#                            rotate_cols=False,
#                            minimize=False)

latex_table = df_f1.to_latex(
    float_format="%.2f",
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    caption=f'F1 score on synthetic data',
    label='tab:synth_metrics',
    position='htb'
)

print(latex_table)
