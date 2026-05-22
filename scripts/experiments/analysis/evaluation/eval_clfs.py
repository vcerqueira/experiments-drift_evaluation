import pandas as pd

PARAM_SETTINGS = ['default', 'hypertuned']
DRIFT_TYPES1 = ['ABRUPT', 'GRADUAL']
DRIFT_TYPES2 = ['x_permutations', 'y_swaps', 'y_prior_skip', 'x_exceed_skip']
CLASSIFIER_LIST = ['ARF','HoeffdingTree','NaiveBayes','OnlineBagging','OzaBoost']

RESULTS_DIR = 'assets/results/real_clf'

NAME_MAPPING = {'GeometricMovingAverage': 'GMA',
                    'EWMAChart': 'EWMA',
                    'HDDMAverage': 'HDDMA',
                    'HDDMWeighted': 'HDDMW',
                    'PageHinkley': 'PH',
                    'ABCDx': 'ABCD(X)', }

metric = 'f1'

all_results = {}
for drift_type2 in DRIFT_TYPES2:

    for learner in CLASSIFIER_LIST:

        file_path = f'{RESULTS_DIR}/Electricity,{drift_type2},ABRUPT,{learner},results.csv'

        df = pd.read_csv(file_path)
        df = df.set_index('Unnamed: 0')
        df.index.name = 'Detector'

        if df.shape[0] == 0:
            continue

        df = df.rename(index=NAME_MAPPING)

        df = df[metric].round(3)

        if df is None:
            continue

        all_results[drift_type2, learner] = df

all_results_df = pd.DataFrame(all_results).T.reset_index()
all_results_df.rename(columns={
    'level_0': 'Type',
    'level_1': 'Learner',
}, inplace=True)

all_results_df['Type'] = all_results_df['Type'].map({
    'x_permutations': 'X-Perm',
    'y_swaps': 'Y-Swaps',
    'y_prior_skip': 'Y-Prior',
    'x_exceed_skip': 'X-Exceed'
})

s = all_results_df.query(f'Type=="Y-Swaps"').drop(columns=['Type']).set_index('Learner').T

print(s)

print(s.round(2).astype(str).to_latex(caption='cap',label='tab:clfs'))



