import pandas as pd

results = {}
for stream in ['Agrawal', 'STAGGER', 'SEA']:
    results[stream] = pd.read_csv(f'assets/results/{stream},ABRUPT,ARF,POINT.csv').set_index('Unnamed: 0')['f1']

df = pd.DataFrame(results)
print(df.mean(axis=1).sort_values())

df.rank(axis=1).mean()
df.rank(ascending=False).mean(axis=1).sort_values()
