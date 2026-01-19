import warnings
from typing import Optional

import pandas as pd

from src.streams.config import MAX_DELAY

dataset_list = [*MAX_DELAY]

warnings.filterwarnings('ignore', category=FutureWarning)


class DataReader:
    RESULTS_DIR = 'assets/results/real'

    NAME_MAPPING = {'GeometricMovingAverage': 'GMA',
                    'EWMAChart': 'EWMA',
                    'HDDMAverage': 'HDDMA',
                    'HDDMWeighted': 'HDDMW',
                    'PageHinkley': 'PH',
                    'ABCDx': 'ABCD(X)', }

    @classmethod
    def get_real_results(cls,
                         dataset: str,
                         learner: str,
                         drift_type: str,
                         drift_abruptness: str,
                         param_setting: str,
                         metric: Optional[str] = None,
                         round_to: Optional[int] = None):

        file_path = f'{cls.RESULTS_DIR}/{dataset},{drift_type},{drift_abruptness},{param_setting},results.csv'

        df = pd.read_csv(file_path)
        df = df.set_index('Unnamed: 0')
        df.index.name = 'Detector'

        if df.shape[0] == 0:
            return None

        df = df.rename(index=cls.NAME_MAPPING)

        if metric is not None:
            assert metric in df.columns, f"Metric '{metric}' not found in DataFrame columns."
            df = df[metric]

        if round_to is not None:
            df = df.round(round_to)

        return df

    @classmethod
    def read_all_real_results(cls, metric: str = 'f1', round_to=3):
        PARAM_SETTINGS = ['default', 'hypertuned']
        DRIFT_TYPES1 = ['ABRUPT', 'GRADUAL']
        DRIFT_TYPES2 = ['x_permutations', 'y_swaps', 'y_prior_skip', 'x_exceed_skip']

        all_results = {}
        for ds in dataset_list:
            for param in PARAM_SETTINGS:
                for drift_type1 in DRIFT_TYPES1:
                    for drift_type2 in DRIFT_TYPES2:
                        df = cls.get_real_results(
                            dataset=ds,
                            learner='HoeffdingTree',
                            drift_type=drift_type2,
                            drift_abruptness=drift_type1,
                            param_setting=param,
                            metric=metric,
                            round_to=round_to
                        )

                        if df is None:
                            continue

                        all_results[ds, drift_type1, drift_type2, param] = df

        all_results_df = pd.DataFrame(all_results).T.reset_index()
        all_results_df.rename(columns={
            'level_0': 'Dataset',
            'level_1': 'Mode',
            'level_2': 'Type',
            'level_3': 'Params'
        }, inplace=True)

        all_results_df['Type'] = all_results_df['Type'].map({
            'x_permutations': 'X-Perm',
            'y_swaps': 'Y-Swaps',
            'y_prior_skip': 'Y-Prior',
            'x_exceed_skip': 'X-Exceed'
        })

        all_results_df['Params'] = all_results_df['Params'].map({'default': 'Default', 'hypertuned': 'Optimized'})

        return all_results_df


def average_rank(df):
    df_m = df.melt(
        id_vars=['Type', 'Dataset'],
        var_name='Detector',
        value_name='value'
    )

    ranks = df_m.groupby(['Type', 'Dataset'])['value'].rank(ascending=False, method='average')
    df_m = df_m.assign(Rank=ranks)

    avg_ranks = df_m.groupby(['Type', 'Detector'])['Rank'].mean().unstack()
    return avg_ranks.T


def prep_latex_tab(df,
                   minimize: bool = False,
                   rotate_cols: bool = False,
                   rotate_index: bool = False,
                   round_to: int = None):
    """
    Formats a DataFrame for LaTeX tables with highlighting for best and second-best values.
    
    This function takes a DataFrame of numerical values and formats it for LaTeX output,
    automatically highlighting the best value (in bold) and second-best value (underlined)
    in each row.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing numerical values to format
        minimize (bool, optional): If True, lower values are considered better.
                                  If False, higher values are considered better.
                                  Defaults to False.
        rotate_cols (bool, optional): If True, column headers will be rotated 90 degrees
                                     in the LaTeX output. Defaults to False.
    
    Returns:
        pd.DataFrame: DataFrame with LaTeX formatting applied to values
    
    Raises:
        ValueError: If any row contains fewer than 2 unique values
    """
    formatted_df = df.copy()

    if rotate_cols:
        formatted_df.columns = [f'\\rotatebox{{90}}{{{col}}}' for col in formatted_df.columns]

    if rotate_index:
        if isinstance(formatted_df.index, pd.MultiIndex):
            new_levels = []
            for level in formatted_df.index.levels:
                formatted_level = [f'\\rotatebox{{90}}{{{str(label)}}}' for label in level]
                new_levels.append(formatted_level)

            formatted_df.index = formatted_df.index.set_levels(new_levels)
        else:
            formatted_df.index = [f'\\rotatebox{{90}}{{{str(idx)}}}' for idx in formatted_df.index]

    formatted_rows = []
    for _, row in formatted_df.T.iterrows():
        top_2 = row.sort_values(ascending=minimize).unique()[:2]
        if len(top_2) < 2:
            raise ValueError('Row must contain at least two unique values')

        best_value = row[row == top_2[0]].values[0]
        second_best_value = row[row == top_2[1]].values[0]

        row[row == top_2[0]] = f'\\textbf{{{best_value}}}'
        row[row == top_2[1]] = f'\\underline{{{second_best_value}}}'

        formatted_rows.append(row)

    result_df = pd.DataFrame(formatted_rows).T

    if round_to is not None:
        result_df = result_df.round(round_to)

    result_df = result_df.astype(str)

    return result_df
