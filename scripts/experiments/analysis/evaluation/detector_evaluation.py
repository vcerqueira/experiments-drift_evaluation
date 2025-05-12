import pandas as pd
import plotnine as p9

from src.misc import DataReader, prep_latex_tab

# DATASETS = ['Electricity', 'Covtype']
# PARAM_SETTINGS = ['default', 'hypertuned']
# DRIFT_TYPES1 = ['ABRUPT', 'GRADUAL']
# DRIFT_TYPES2 = ['x_permutations', 'y_swaps', 'y_prior_skip', 'x_exceed_skip']
#
# r = DataReader.get_real_results(
#     dataset='Electricity',
#     learner='HoeffdingTree',
#     drift_type='x_permutations',
#     drift_abruptness='ABRUPT',
#     param_setting='default',
#     metric='f1',
#     round_to=3
# )

r = DataReader.read_all_real_results(metric='f1', round_to=3)


# --- hypertuning v default ---

r.query('Mode == "ABRUPT"').drop(columns=['Mode'])


# r.groupby(['Mode','Params']).mean(numeric_only=True).T
r_melt = r.drop(columns=['Dataset', 'Type']).melt(['Mode','Params'])

p = (
    p9.ggplot(r_melt, p9.aes(x='Detector', y='value', fill='Params'))
    + p9.geom_bar(stat='identity', position='dodge')
    + p9.facet_wrap('~ Mode', nrow=2)
    + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
    + p9.labs(title='',x='',y='F1')
)

# To save the plot to a file
p.save("detector_comparison.pdf", width=10, height=6)

### ---- latex table ---- ###

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
