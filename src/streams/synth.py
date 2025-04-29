import copy

import numpy as np
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
from capymoa.stream.generator import SEA, AgrawalGenerator, STAGGERGenerator


class CustomDriftStream:
    GENERATORS = {
        'SEA': SEA,
        'Agrawal': AgrawalGenerator,
        'STAGGER': STAGGERGenerator,
    }

    FUNCTION_LIST = {
        'SEA': [1, 2, 3, 4],
        'Agrawal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'STAGGER': [1, 2, 3],
    }

    def __init__(self,
                 generator: str,
                 n_drifts: int = 50,
                 drift_every_n: int = 2000,
                 drift_width: int = 0,
                 seed: int = 123):

        self.generator = generator
        self.n_drifts = n_drifts
        self.drift_every_n = drift_every_n
        self.drift_width = drift_width
        self.seed = seed

        self.init_func = 1

    def create_stream(self):

        np.random.seed(self.seed)

        current_func = self.init_func

        drift_point = copy.deepcopy(self.drift_every_n)

        if self.generator == 'SEA':
            stream_ = [self.GENERATORS[self.generator](function=current_func)]
        else:
            stream_ = [self.GENERATORS[self.generator](classification_function=current_func)]

        if self.drift_width > 0:
            stream_.append(GradualDrift(position=drift_point, width=self.drift_width))
        else:
            stream_.append(AbruptDrift(position=drift_point))

        for i in range(1, self.n_drifts):
            funcs_i = [j for j in self.FUNCTION_LIST[self.generator] if j != current_func]

            current_func = np.random.choice(funcs_i)
            drift_point += self.drift_every_n + self.drift_width

            if self.generator == 'SEA':
                stream_.append(self.GENERATORS[self.generator](function=current_func))
            else:
                stream_.append(self.GENERATORS[self.generator](classification_function=current_func))

            if self.drift_width > 0:
                stream_.append(GradualDrift(position=drift_point, width=self.drift_width))
            else:
                stream_.append(AbruptDrift(position=drift_point))

        if self.generator == 'SEA':
            stream_.append(self.GENERATORS[self.generator](function=current_func))
        else:
            stream_.append(self.GENERATORS[self.generator](classification_function=current_func))

        drift_stream = DriftStream(stream=stream_)

        return drift_stream
