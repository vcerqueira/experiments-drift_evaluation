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

STREAM_MEDIANS = {'covtype-weka.filters.unsupervised.attribute.Normalize-S1.0-T0.0': {'Aspect': 108.0,
                                                                                      'Elevation': 2900.0,
                                                                                      'Hillshade_3pm': 139.0,
                                                                                      'Hillshade_9am': 222.0,
                                                                                      'Hillshade_Noon': 227.0,
                                                                                      'Horizontal_Distance_To_Fire_Points': 2719.0,
                                                                                      'Horizontal_Distance_To_Hydrology': 201.0,
                                                                                      'Horizontal_Distance_To_Roadways': 3469.0,
                                                                                      'Slope': 10.0,
                                                                                      'Vertical_Distance_To_Hydrology': 20.0},
                  'electricity-weka.filters.unsupervised.attribute.Normalize-S1.0-T0.0-weka.filters.unsupervised.attribute.ReplaceMissingValues': {
                      'nswdemand': 0.4631805,
                      'nswprice': 0.075117,
                      'period': 0.489362,
                      'transfer': 0.414912,
                      'vicdemand': 0.422915,
                      'vicprice': 0.003467},
                  'sensor-all-56': {'humidity': 39.2123,
                                    'light': 158.24,
                                    'rcdminutes': 728.0,
                                    'temperature': 22.4678,
                                    'voltage': 2.52732}}
