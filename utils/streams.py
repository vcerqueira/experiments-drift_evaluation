from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
from capymoa.stream.generator import SEA

stream_sea_abrupt = DriftStream(
    stream=[
        SEA(function=1),
        AbruptDrift(position=2000),
        SEA(function=3),
        AbruptDrift(position=4000),
        SEA(function=2),
        AbruptDrift(position=6000),
        SEA(function=4),
        AbruptDrift(position=8000),
        SEA(function=1),
    ]
)

stream_sea_gradual = DriftStream(
    stream=[
        SEA(function=1),
        GradualDrift(position=2000, width=1000),
        SEA(function=3),
        GradualDrift(position=4000, width=1000),
        SEA(function=2),
        GradualDrift(position=6000, width=1000),
        SEA(function=4),
        GradualDrift(position=8000, width=1000),
        SEA(function=1),
    ]
)
