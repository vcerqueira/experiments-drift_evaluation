import pandas as pd

mode = 'gradual'

point = pd.read_csv(f'assets/{mode},OnlineBagging,point.csv').set_index('Unnamed: 0')
window = pd.read_csv(f'assets/{mode},OnlineBagging,window.csv').set_index('Unnamed: 0')

# point['f1']
# window['f1']

df = pd.concat([point['f1'], window['f1']], axis=1)
df.columns = ['point', 'window']
print(df)