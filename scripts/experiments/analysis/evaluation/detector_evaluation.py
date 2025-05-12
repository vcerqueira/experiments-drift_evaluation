import pandas as pd
import numpy as np
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

THEME = p9.theme_538(base_family='Palatino', base_size=12) + \
        p9.theme(plot_margin=.025,
                 panel_background=p9.element_rect(fill='white'),
                 plot_background=p9.element_rect(fill='white'),
                 legend_box_background=p9.element_rect(fill='white'),
                 strip_background=p9.element_rect(fill='white'),
                 legend_background=p9.element_rect(fill='white'),
                 axis_text_x=p9.element_text(size=9, angle=0),
                 axis_text_y=p9.element_text(size=9),
                 legend_title=p9.element_blank())

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
        + THEME
        + p9.theme(axis_text_x=p9.element_text(angle=30),
                   legend_position='top',
                   strip_background=p9.element_text(color='lightgrey'))
        + p9.labs(title='', x='', y='F1')
)

p.save("detector_comparison.pdf", width=10, height=6)

########################################################
# --- overall tables ---
########################################################

r_ht = r.query('Params == "Optimized"').drop(columns=['Params'])
r_ht_abp = r_ht.query('Mode == "ABRUPT"').drop(columns=['Mode'])
r_ht_grd = r_ht.query('Mode == "GRADUAL"').drop(columns=['Mode'])

r_ht_idx = r_ht.set_index(['Type', 'Mode', 'Dataset']).sort_index()
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
        + THEME
        + p9.theme(axis_text_x=p9.element_text(size=13),
                   strip_background=p9.element_text(color='lightgrey'))
        + p9.labs(title='', x='', y='F1')
)

p2.save("detector_comparison2.pdf", width=12, height=6)

r_ht_grd_melt = r_ht_grd.melt(['Type', 'Dataset'])

p3 = (
        p9.ggplot(r_ht_grd_melt, p9.aes(x='Dataset', y='value', fill='Detector'))
        + p9.geom_bar(stat='identity', position='dodge')
        + p9.facet_wrap('~ Type', nrow=2)
        + THEME
        + p9.theme(axis_text_x=p9.element_text(size=13),
                   strip_background=p9.element_text(color='lightgrey'))
        + p9.labs(title='', x='', y='F1')
)

p3.save("detector_comparison3.pdf", width=12, height=6)

########################################################
# --- abruptness ---
########################################################


r_melt = r_ht.drop(columns=['Dataset', 'Type']).melt(['Mode'])

p4 = (
        p9.ggplot(r_melt, p9.aes(x='Detector', y='value', fill='Mode'))
        + p9.geom_boxplot(position='dodge')
        + THEME
        + p9.theme(axis_text_x=p9.element_text(angle=30),
                   legend_position='top',
                   strip_background=p9.element_text(color='lightgrey'))
        + p9.labs(title='', x='', y='F1')
)

p4.save("detector_comparison4.pdf", width=10, height=5)

########################################################
# --- f1 vs far ---
########################################################

r_far = DataReader.read_all_real_results(metric='far', round_to=3)
r_far = r_far.query('Params == "Optimized"').drop(columns=['Params'])

r_far_melt = r_far.melt(['Dataset', 'Mode', 'Type']).rename(columns={'value':'FAR'})
r_f1_melt = r_ht.melt(['Dataset', 'Mode', 'Type']).rename(columns={'value':'F1'})

r_both = r_f1_melt.merge(r_far_melt, on=['Dataset', 'Mode', 'Type', 'Detector', ])



p5 = (
    p9.ggplot(r_both, p9.aes(x='np.log(FAR+1)', y='F1'))
    + p9.geom_point(alpha=0.7)
    + p9.facet_wrap('~ Type', nrow=2)
    + THEME
    + p9.theme(legend_position='top',
               strip_background=p9.element_text(color='lightgrey'))
    + p9.labs(title='', x='log FAR', y='F1')
)

p5.save("detector_comparison5.pdf", width=12, height=8)


########################################################
# --- f1 vs mdt ---
########################################################

r_mdt = DataReader.read_all_real_results(metric='mdt', round_to=3)
r_mdt = r_mdt.query('Params == "Optimized"').drop(columns=['Params'])

r_mdt_melt = r_mdt.melt(['Dataset', 'Mode', 'Type']).rename(columns={'value':'MDT'})
r_f1_melt = r_ht.melt(['Dataset', 'Mode', 'Type']).rename(columns={'value':'F1'})

r_both_mdt = r_f1_melt.merge(r_mdt_melt, on=['Dataset', 'Mode', 'Type', 'Detector', ])

p6 = (
    p9.ggplot(r_both_mdt, p9.aes(x='np.log(MDT+1)', y='F1'))
    + p9.geom_point(alpha=0.7)
    + p9.facet_wrap('~ Type', nrow=2)
    + THEME
    + p9.theme(legend_position='top',
               strip_background=p9.element_text(color='lightgrey'))
    + p9.labs(title='', x='log FAR', y='F1')
)

p6.save("detector_comparison6.pdf", width=12, height=8)


########################################################
# --- avg ranks ---
########################################################
model_names = r_mdt.select_dtypes(include='number').columns.tolist()

# r_mdt.groupby(['Dataset','Mode','Type']).apply(lambda x: x[model_names].rank(axis=1).mean())

avg_rank_mdt = r_mdt[model_names].rank(axis=1, ascending=True, na_option='bottom').mean()
avg_rank_f1 = r_ht[model_names].rank(axis=1, ascending=False, na_option='bottom').mean()
avg_rank_far = r_far[model_names].rank(axis=1, ascending=True, na_option='bottom').mean()

avg_ranks = pd.concat([avg_rank_f1, avg_rank_mdt, avg_rank_far],axis=1)
avg_ranks.columns = ['F1','MDT','FAR']

p7 = (
    p9.ggplot(avg_ranks.reset_index(),
              p9.aes(x='FAR', y='F1', size='MDT', label='Detector'))
    + p9.geom_point(alpha=0.7)
    + p9.geom_text(nudge_y=0.5, size=8)
    + THEME
    + p9.theme(legend_position='top',
               legend_title=p9.element_text(text='Avg. Rank (MDT)'),
               strip_background=p9.element_text(color='lightgrey'))
    + p9.labs(title='', 
              x='Avg. Rank (FAR)',
              y='Avg. Rank (F1)',
              size='Avg. Rank (MDT)')
    + p9.scale_size_continuous(range=(3, 12))
)

p7.save("detector_comparison7.pdf", width=10, height=8)


