import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.misc import DataReader

r = DataReader.read_all_real_results(metric='f1', round_to=3)

########################################################
# wilcoxon sr test
########################################################

r_ht = r.query('Params == "Optimized"').drop(columns=['Params'])
r_ht_abp = r_ht.query('Mode == "ABRUPT"').drop(columns=['Mode'])
r_ht_grd = r_ht.query('Mode == "GRADUAL"').drop(columns=['Mode'])


def compare_to_control(df, control='SEED', alpha=0.001, correction='bonferroni'):
    """
    Compare a control method (e.g., best performer) against all others
    using Wilcoxon signed-rank test with multiple comparison correction.
    
    Parameters:
    -----------
    df : DataFrame with detector columns
    control : name of the control/best method
    alpha : significance level
    correction : 'holm', 'bonferroni', 'fdr_bh' (Benjamini-Hochberg)
    
    Returns:
    --------
    DataFrame with test results
    """
    detectors = df.select_dtypes(include='number').columns.tolist()
    others = [d for d in detectors if d != control]

    control_scores = df[control].values

    results = []
    for detector in others:
        other_scores = df[detector].values

        # Wilcoxon signed-rank test (two-sided by default)
        # Use alternative='greater' if you specifically test control > other
        try:
            stat, p_value = wilcoxon(control_scores, other_scores, alternative='greater')
        except ValueError:
            # All differences are zero
            stat, p_value = 0, 1.0

        n = len(control_scores)
        wins = np.sum(control_scores > other_scores)
        losses = np.sum(control_scores < other_scores)
        effect_size = (wins - losses) / n

        results.append({
            'Detector': detector,
            'W-statistic': stat,
            'p-value': p_value,
            'Wins': wins,
            'Losses': losses,
            'Ties': n - wins - losses,
            'Effect Size': effect_size
        })

    results_df = pd.DataFrame(results)

    _, p_adjusted, _, _ = multipletests(results_df['p-value'], alpha=alpha, method=correction)
    results_df['p-adjusted'] = p_adjusted
    results_df['Significant'] = results_df['p-adjusted'] < alpha

    results_df = results_df.sort_values('p-value')

    print(results_df.to_string(index=False))

    n_sig = results_df['Significant'].sum()
    print(f"\n{control} is significantly better than {n_sig}/{len(others)} methods")

    return results_df


print("=" * 80)
print("ABRUPT: SEED vs All Others")
print("=" * 80)
wilcox_abrupt = compare_to_control(r_ht_abp, control='SEED')

print("\n" + "=" * 80)
print("GRADUAL: SEED vs All Others")
print("=" * 80)
wilcox_gradual = compare_to_control(r_ht_grd, control='SEED')
