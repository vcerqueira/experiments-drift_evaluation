import pandas as pd

mode = 'abrupt'

# point = pd.read_csv(f'assets/{mode},OnlineBagging,point.csv').set_index('Unnamed: 0')
# window = pd.read_csv(f'assets/{mode},ARF,window.csv').set_index('Unnamed: 0')
df = pd.read_csv(f'assets/Electricity,ABRUPT@Y,HoeffdingTree,POINT.csv').set_index('Unnamed: 0')

# point['f1']
# window['f1']

# df = pd.concat([point['f1'], window['f1']], axis=1)
# df = window
# df.columns = ['point', 'window']
print(df)
print(df['f1'].sort_values())
