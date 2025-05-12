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



########################################################
# --- hypertuning v default ---
########################################################

# r.groupby(['Mode','Params']).mean(numeric_only=True).T
r_melt = r.drop(columns=['Dataset', 'Type']).melt(['Mode', 'Params'])

# p = (
#         p9.ggplot(r_melt, p9.aes(x='Detector', y='value', fill='Params'))
#         + p9.geom_bar(stat='identity', position='dodge')
#         + p9.facet_wrap('~ Mode', nrow=2)
#         + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
#         + p9.labs(title='', x='', y='F1')
# )

p = (
        p9.ggplot(r_melt, p9.aes(x='Detector', y='value', fill='Params'))
        + p9.geom_boxplot(position='dodge')
        + p9.facet_wrap('~ Mode', nrow=2)
        + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
        + p9.labs(title='', x='', y='F1')
)

p.save("detector_comparison.pdf", width=10, height=6)

########################################################
# --- overall tables ---
########################################################

r_ht = r.query('Params == "hypertuned"').drop(columns=['Params'])
r_ht_abp = r_ht.query('Mode == "ABRUPT"').drop(columns=['Mode'])
r_ht_grd = r_ht.query('Mode == "GRADUAL"').drop(columns=['Mode'])

r_ht_idx = r_ht.set_index(['Type', 'Mode','Dataset']).sort_index()
print(r_ht_idx)

r_ht_abp_idx = r_ht_abp.set_index(['Type', 'Dataset']).sort_index()
print(r_ht_abp_idx)
r_ht_grd_idx = r_ht_grd.set_index(['Type', 'Dataset']).sort_index()
print(r_ht_grd_idx)

r_ht_abp_idx_tab = prep_latex_tab(r_ht_abp_idx,
                                  minimize=False,
                                  rotate_index=True,
                                  rotate_cols=False)

latex_table_abr = r_ht_abp_idx_tab.T.to_latex(
    float_format="%.3f",
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    caption='Performance metrics of drift detectors across different datasets',
    label='tab:combined_metrics',
    position='htbp'
)

print(latex_table_abr)

r_ht_grd_idx_tab = prep_latex_tab(r_ht_grd_idx,
                                  minimize=False,
                                  rotate_index=True,
                                  rotate_cols=False)

latex_table_grd = r_ht_grd_idx_tab.T.to_latex(
    float_format="%.3f",
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    caption='Performance metrics of drift detectors across different datasets',
    label='tab:combined_metrics',
    position='htbp'
)

print(latex_table_grd)


r_ht_abp_melt = r_ht_abp.melt(['Type', 'Dataset'])

p2 = (
        p9.ggplot(r_ht_abp_melt, p9.aes(x='Dataset', y='value', fill='Detector'))
        + p9.geom_bar(stat='identity', position='dodge')
        + p9.facet_wrap('~ Type', nrow=2)
        + p9.theme_538()
        + p9.theme(axis_text_x=p9.element_text(size=13))
        + p9.labs(title='', x='', y='F1')
)

p2.save("detector_comparison2.pdf", width=12, height=10)

r_ht_grd_melt = r_ht_grd.melt(['Type', 'Dataset'])

p3 = (
        p9.ggplot(r_ht_grd_melt, p9.aes(x='Dataset', y='value', fill='Detector'))
        + p9.geom_bar(stat='identity', position='dodge')
        + p9.facet_wrap('~ Type', nrow=2)
        + p9.theme_538()
        + p9.theme(axis_text_x=p9.element_text(size=13))
        + p9.labs(title='', x='', y='F1')
)

p3.save("detector_comparison3.pdf", width=12, height=10)

########################################################
# --- abruptness ---
########################################################




### ---- latex table ---- ###

