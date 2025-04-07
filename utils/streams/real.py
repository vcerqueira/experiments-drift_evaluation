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

STREAM_MEDIANS = {'covtype-weka.filters.unsupervised.attribute.Normalize-S1.0-T0.0': {'Aspect': 0.352778,
                                                                                      'Elevation': 0.568784,
                                                                                      'Hillshade_3pm': 0.562992,
                                                                                      'Hillshade_9am': 0.858268,
                                                                                      'Hillshade_Noon': 0.889764,
                                                                                      'Horizontal_Distance_To_Fire_Points': 0.238394,
                                                                                      'Horizontal_Distance_To_Hydrology': 0.156049,
                                                                                      'Horizontal_Distance_To_Roadways': 0.280596,
                                                                                      'Slope': 0.19697,
                                                                                      'Vertical_Distance_To_Hydrology': 0.262274},
                  'electricity-weka.filters.unsupervised.attribute.Normalize-S1.0-T0.0-weka.filters.unsupervised.attribute.ReplaceMissingValues': {
                      'date': 0.456329,
                      'nswdemand': 0.44369250000000005,
                      'nswprice': 0.048652,
                      'period': 0.5,
                      'transfer': 0.414912,
                      'vicdemand': 0.422915,
                      'vicprice': 0.003467}}
