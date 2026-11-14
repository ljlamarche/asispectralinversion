"""Instrument calibration constants shared by mapping and inversion."""

FRAME_INTERVAL_SECONDS_GREEN = 0.3
FRAME_INTERVAL_SECONDS_RED = 0.9
FRAME_INTERVAL_SECONDS_BLUE = 0.9

GREEN_RAYLEIGH_SECONDS_PER_COUNT = {"ARV": 64.0, "VEE": 35.0, "BVR": 57.0}
RED_RAYLEIGH_SECONDS_PER_COUNT = {"ARV": 40.0, "VEE": 69.5, "BVR": 30.2}
BLUE_RAYLEIGH_SECONDS_PER_COUNT = {"ARV": 35.5, "VEE": 17.0, "BVR": 1.7}
RAYLEIGH_SECONDS_PER_COUNT_BY_COLOR = {
    "green": GREEN_RAYLEIGH_SECONDS_PER_COUNT,
    "red": RED_RAYLEIGH_SECONDS_PER_COUNT,
    "blue": BLUE_RAYLEIGH_SECONDS_PER_COUNT,
}
EXPOSURE_TIME_S_BY_COLOR = {
    "green": FRAME_INTERVAL_SECONDS_GREEN,
    "red": FRAME_INTERVAL_SECONDS_RED,
    "blue": FRAME_INTERVAL_SECONDS_BLUE,
}


def calibration_factor(site, color):
    return RAYLEIGH_SECONDS_PER_COUNT_BY_COLOR.get(str(color).lower(), {}).get(
        str(site).upper()
    )


def exposure_time_s(color):
    return EXPOSURE_TIME_S_BY_COLOR.get(str(color).lower())
