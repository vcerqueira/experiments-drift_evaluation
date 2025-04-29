import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class DataReader:
    STREAMS = ['Agrawal',
               'STAGGER',
               'SEA']
    CLASSIFIERS = ['HoeffdingTree',
                   'ARF',
                   'NaiveBayes']

    @classmethod
    def get_synth_results(cls,
                          metric: str,
                          round_to=None,
                          stream_list=STREAMS,
                          learners=CLASSIFIERS):

        results = []
        for stream in stream_list:
            for classifier in learners:
                df = pd.read_csv(f'assets/results/{stream},ABRUPT,{classifier}.csv', index_col='Unnamed: 0')

                # df = df.drop('ABCDx').drop(columns=['error'])

                df_result = df[metric]

                df_result.name = (stream, classifier)

                results.append(df_result)

        df_all = pd.concat(results, axis=1).T.reset_index().reset_index(drop=True)
        df_all = df_all.rename(columns={'level_0': 'Stream', 'level_1': 'Classifier'})

        if round_to is not None:
            df_all = df_all.round(round_to)

        return df_all


def prep_latex_tab(df, minimize: bool = False, rotate_cols: bool = False):
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

    formatted_rows = []
    for _, row in formatted_df.iterrows():
        top_2 = row.sort_values(ascending=minimize).unique()[:2]
        if len(top_2) < 2:
            raise ValueError('Row must contain at least two unique values')

        best_value = row[row == top_2[0]].values[0]
        second_best_value = row[row == top_2[1]].values[0]

        row[row == top_2[0]] = f'\\textbf{{{best_value}}}'
        row[row == top_2[1]] = f'\\underline{{{second_best_value}}}'

        formatted_rows.append(row)

    result_df = pd.DataFrame(formatted_rows).astype(str)

    return result_df
