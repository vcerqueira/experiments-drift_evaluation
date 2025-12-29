from pprint import pprint

from src.streams.read_from_df import StreamFromDF

# https://sites.google.com/view/uspdsrepository
DATASETS = [
    'data/Covtype.csv',
    'data/Electricity.csv',
    'data/Asfault.csv',
    'data/GasSensorArray.csv',
    'data/NOAA.csv',
    'data/Posture.csv',
    'data/Rialto.csv',
]

stream_stats = {}
for stream_fp in DATASETS:
    print(stream_fp)
    stream_name = StreamFromDF.get_stream_name(stream_fp)
    df = StreamFromDF.read_csv(stream_fp, shuffle=False, as_np_stream=False)
    df = df.drop(columns=['target'])

    stream_stats[stream_name] = df.median().to_dict()

pprint(stream_stats)
