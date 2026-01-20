from src.misc import DataReader, prep_latex_tab

metric = 'recall'

r = DataReader.read_all_real_results(metric=metric, round_to=2)

r_ht = r.query('Params == "Optimized"').drop(columns=['Params'])
r_ht_abp = r_ht.query('Mode == "ABRUPT"').drop(columns=['Mode'])
r_ht_grd = r_ht.query('Mode == "GRADUAL"').drop(columns=['Mode'])

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
    caption=f'Average {metric} score of drift detectors across different datasets for abrupt drifts',
    label=f'tab:app_{metric}_abr',
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
    caption=f'Average {metric} score of drift detectors across different datasets for gradual drifts',
    label=f'tab:app_{metric}_grd',
    position='htbp'
)

print(grd_mean_tabl)
