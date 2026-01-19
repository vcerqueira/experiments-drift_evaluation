import pandas as pd
import numpy as np
import plotnine as p9

from src.misc import DataReader, prep_latex_tab, average_rank

OUTPUT_DIR = 'assets/results/outputs/'

THEME = p9.theme_538(base_family='Palatino', base_size=14) + \
        p9.theme(plot_margin=.015,
                 panel_background=p9.element_rect(fill='white'),
                 plot_background=p9.element_rect(fill='white'),
                 legend_box_background=p9.element_rect(fill='white'),
                 strip_background=p9.element_rect(fill='white'),
                 legend_background=p9.element_rect(fill='white'),
                 axis_text_x=p9.element_text(size=15, angle=30),
                 axis_text_y=p9.element_text(size=15),
                 legend_title=p9.element_blank())

r = DataReader.read_all_real_results(metric='f1', round_to=3)

########################################################
# --- hypertuning v default ---
########################################################

# r.groupby(['Mode','Params']).mean(numeric_only=True).T
r_melt = r.drop(columns=['Dataset', 'Type']).melt(['Mode', 'Params'])

p = (
        p9.ggplot(r_melt, p9.aes(x='Detector', y='value', fill='Params'))
        + p9.geom_boxplot(position='dodge')
        + p9.facet_wrap('~ Mode', nrow=2)
        + THEME
        + p9.theme(legend_position='top',
                   strip_background=p9.element_text(color='lightgrey'))
        + p9.labs(title='', x='', y='F1')
)

p.save(f"{OUTPUT_DIR}/plot1_hypertuning.pdf", width=10, height=8)

########################################################
# --- abruptness ---
########################################################


r_ht = r.query('Params == "Optimized"').drop(columns=['Params'])
r_ht_abp = r_ht.query('Mode == "ABRUPT"').drop(columns=['Mode'])
r_ht_grd = r_ht.query('Mode == "GRADUAL"').drop(columns=['Mode'])

r_melt = r_ht.drop(columns=['Dataset', 'Type']).melt(['Mode'])

p4 = (
        p9.ggplot(r_melt, p9.aes(x='Detector', y='value', fill='Mode'))
        + p9.geom_boxplot(position='dodge')
        + THEME
        + p9.theme(legend_position='top',
                   strip_background=p9.element_text(color='lightgrey'))
        + p9.labs(title='', x='', y='F1')
)

p4.save(f"{OUTPUT_DIR}/plot4_abrupt_gradual.pdf", width=10, height=5)

########################################################
# --- overall tables ---
########################################################

ar_abp_df = average_rank(r_ht_abp).round(1)
ar_abp_df_tab = prep_latex_tab(ar_abp_df,
                               minimize=True,
                               rotate_index=False,
                               rotate_cols=False)

ar_abp_df_tabl = ar_abp_df_tab.to_latex(
    float_format="%.3f",
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    caption='Average rank of drift detectors across different datasets for abrupt drifts',
    label='tab:avgrank_abr',
    position='htbp'
)
print(ar_abp_df_tabl)

ar_grd_df = average_rank(r_ht_grd).round(1)

ar_grd_df_tab = prep_latex_tab(ar_grd_df,
                               minimize=True,
                               rotate_index=False,
                               rotate_cols=False)

ar_grd_df_tabl = ar_grd_df_tab.to_latex(
    float_format="%.3f",
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    caption='Average rank of drift detectors across different datasets for gradual drifts',
    label='tab:avgrank_abr',
    position='htbp'
)
print(ar_grd_df_tabl)

# Avg scores

abp_mean = r_ht_abp.groupby('Type').mean(numeric_only=True).T
grd_mean = r_ht_grd.groupby('Type').mean(numeric_only=True).T

abp_mean_tab = prep_latex_tab(abp_mean.round(2),
                              minimize=False,
                              rotate_index=False,
                              rotate_cols=False)

abp_mean_tabl = abp_mean_tab.to_latex(
    float_format="%.3f",
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    caption='Average F1 detection score of drift detectors across different datasets for abrupt drifts',
    label='tab:combined_metrics',
    position='htbp'
)

print(abp_mean_tabl)

grd_mean_tab = prep_latex_tab(grd_mean.round(2),
                              minimize=False,
                              rotate_index=False,
                              rotate_cols=False)

grd_mean_tabl = grd_mean_tab.to_latex(
    float_format="%.3f",
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    caption='Average F1 detection score of drift detectors across different datasets for gradual drifts',
    label='tab:combined_metrics',
    position='htbp'
)

print(grd_mean_tabl)
