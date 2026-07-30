"""Small mission policy helpers without repository-relative data paths."""

import datetime as dt
from dataclasses import dataclass

ROCKET_TIME_WINDOWS = {
    "397": ("101900", "102818"),
    "398": ("101930", "102848"),
    "381": ("070715", "071636"),
    "380": ("083501", "084410"),
}
ROCKET_DEFAULT_DATES = {
    "397": "20260210",
    "398": "20260210",
    "381": "20250202",
    "380": "20250209",
}


@dataclass(frozen=True)
class TrajectoryConfig:
    key: str
    tag: str
    label: str
    path: object


def rocket_launch_datetime(rocket):
    date = ROCKET_DEFAULT_DATES[str(rocket)]
    start = ROCKET_TIME_WINDOWS[str(rocket)][0]
    return dt.datetime.strptime(date + start, "%Y%m%d%H%M%S")


def rocket_time_window_datetimes(rocket):
    date = ROCKET_DEFAULT_DATES[str(rocket)]
    start, end = ROCKET_TIME_WINDOWS[str(rocket)]
    return (
        dt.datetime.strptime(date + start, "%Y%m%d%H%M%S"),
        dt.datetime.strptime(date + end, "%Y%m%d%H%M%S"),
    )


def trajectory_config_tuples(mission, paths=None, date=None):
    if str(mission).upper() != "GNEISS":
        raise ValueError("Portable trajectory configuration currently supports GNEISS")
    paths = paths or {}
    return [
        ("left", "397", "36.397", paths.get("397")),
        ("right", "398", "36.398", paths.get("398")),
    ]
