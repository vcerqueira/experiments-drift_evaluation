import copy
from pprint import pprint

import numpy as np
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
from capymoa.stream.generator import SEA


def create_custom_drift_stream(n_drifts: int,
                               drift_every_n: int = 2000,
                               drift_width: int = 0):
    np.random.seed(123)
    func_l = [1, 2, 3, 4]

    current_func = 1
    drift_point = copy.deepcopy(drift_every_n)

    stream_ = [SEA(function=current_func)]

    if drift_width > 0:
        stream_.append(GradualDrift(position=drift_point, width=drift_width))
    else:
        stream_.append(AbruptDrift(position=drift_point))

    for i in range(1, n_drifts):
        funcs_i = [j for j in func_l if j != current_func]

        current_func = np.random.choice(funcs_i)
        drift_point += drift_every_n + drift_width

        stream_.append(SEA(function=current_func))

        if drift_width > 0:
            stream_.append(GradualDrift(position=drift_point, width=drift_width))
        else:
            stream_.append(AbruptDrift(position=drift_point))

    stream_.append(SEA(function=current_func))

    drift_stream = DriftStream(stream=stream_)

    return drift_stream

# stream_sea_abrupt = create_stream(50, 2000)

# stream_sea_abrupt = DriftStream(
#     stream=[
#         SEA(function=1),
#         AbruptDrift(position=2000),
#         SEA(function=3),
#         AbruptDrift(position=4000),
#         SEA(function=2),
#         AbruptDrift(position=6000),
#         SEA(function=4),
#         AbruptDrift(position=8000),
#         SEA(function=1),
#     ]
# )

# stream_sea_gradual = DriftStream(
#     stream=[
#         SEA(function=1),
#         GradualDrift(position=2000, width=1000),
#         SEA(function=3),
#         GradualDrift(position=4000, width=1000),
#         SEA(function=2),
#         GradualDrift(position=6000, width=1000),
#         SEA(function=4),
#         GradualDrift(position=8000, width=1000),
#         SEA(function=1),
#     ]
# )
