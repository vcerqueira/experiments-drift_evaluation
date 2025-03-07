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
    'Covtype': CovtypeNorm,
    # 'Sensor': Sensor,
}

MAX_DELAY = {
    'Electricity': 2500, 'Covtype': 5000, 'Sensor': 5000,
}

# STABLE_PERIODS = {
#     'Electricity': [],
#     'Covtype': [(224000, 233806), (517400, 526300)],
#     'Sensor': [(7304, 19200),
#                (170504, 193300),
#                (513480, 523900),
#                (536840, 548700),
#                (593512, 609000),
#                (719000, 729700),
#                (808939, 826000),
#                (861300, 874600),
#                (889800, 901500),
#                (955752, 966000),
#                (1068000, 1082800),
#                (1132800, 1142900),
#                (1257000, 1269400),
#                (1291000, 1305600),
#                (1376300, 1389200),
#                (1467000, 1478600),
#                (1631242, 1642100),
#                (1679048, 1690900),
#                (1865640, 1879500),
#                (2098700, 2110900)]
# }
