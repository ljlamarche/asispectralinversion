from pathlib import Path

import numpy as np

from asispectralinversion.config import load_config
from asi_shared.calibration import calibration_factor
from asi_shared.masks import calculate_masks
from asi_shared.trajectory import load_traj_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_example_config_resolves_relative_paths():
    config = load_config(PROJECT_ROOT / "config.example.toml")
    assert config.paths.image_root == PROJECT_ROOT / "data/images"
    assert config.date == "20260210"
    assert config.trajectory_397 == (
        PROJECT_ROOT / "data/trajectories/GNEISS/36397_GPS_Time_Export_01.csv"
    )


def test_shared_calibration_constants():
    assert calibration_factor("VEE", "green") == 35.0
    assert calibration_factor("ARV", "red") == 23.0


def test_small_trajectory_fixture():
    utc, flight, lat, lon, altitude = load_traj_records(
        PROJECT_ROOT / "examples/data/trajectory_397_sample.csv"
    )
    assert np.allclose(utc, [37140.0, 37141.0])
    assert np.allclose(flight, [0.0, 1.0])
    assert lat.shape == lon.shape == altitude.shape == (2,)


def test_overlap_mask_shapes():
    azimuth = np.full((3, 4), 90.0)
    elevation = np.full((3, 4), 30.0)
    first, second = calculate_masks(
        67.0, -146.0, azimuth, elevation,
        66.0, -147.0, azimuth, elevation,
    )
    assert first.shape == second.shape == azimuth.shape
