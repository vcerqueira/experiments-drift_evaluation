from capymoa.datasets import (ElectricityTiny,
                              Bike,
                              CovtypeNorm,
                              Electricity,
                              CovtypeTiny,
                              FriedTiny,
                              Fried,
                              Sensor)

CAPYMOA_DATASETS = {
    'Electricity': Electricity,
    # 'Covtype': CovtypeNorm,
    # 'Sensor': Sensor,
}

MAX_DELAY = {
    'Electricity': 2500, 'Covtype': 5000, 'Sensor': 5000,
}
