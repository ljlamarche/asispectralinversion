"""Portable configuration for notebooks and command-line workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ENV_CONFIG = "ASISPECTRAL_CONFIG"
ENV_PATHS = {
    "image_root": "ASI_IMAGE_ROOT",
    "starmap_root": "ASI_STARMAP_ROOT",
    "glow_root": "ASI_GLOW_ROOT",
    "output_root": "ASI_OUTPUT_ROOT",
    "trajectory_root": "ASI_TRAJECTORY_ROOT",
}


@dataclass(frozen=True)
class PathsConfig:
    image_root: Path
    starmap_root: Path
    glow_root: Path
    output_root: Path
    trajectory_root: Path


@dataclass(frozen=True)
class GneissConfig:
    paths: PathsConfig
    date: str = "20260210"
    start: str = "10:19:00"
    end: str = "10:28:40"
    step_seconds: float = 1.0
    diagnostic_time: str = "10:25:20"
    apex_reference_time: str = "10:24:00"
    target_grid_spacing_km: float = 2.0
    decimation: int = 1
    trajectory_397: Path | None = None
    trajectory_398: Path | None = None


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def default_config_path(project_root: Path | None = None) -> Path:
    root = Path(project_root or Path.cwd()).resolve()
    return Path(os.environ.get(ENV_CONFIG, root / "config.toml")).expanduser()


def load_config(path: str | Path | None = None, project_root: Path | None = None) -> GneissConfig:
    """Load TOML configuration, resolving relative paths beside the config file."""
    project = Path(project_root or Path.cwd()).resolve()
    config_path = Path(path).expanduser() if path else default_config_path(project)
    if not config_path.is_absolute():
        config_path = (project / config_path).resolve()
    if not config_path.exists():
        example = project / "config.example.toml"
        raise FileNotFoundError(
            f"Configuration not found: {config_path}. Copy {example.name} to "
            "config.toml and set your local data paths, or set ASISPECTRAL_CONFIG."
        )

    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    base = config_path.parent
    paths_raw = raw.get("paths", {})

    def configured_path(name: str, default: str) -> Path:
        value = os.environ.get(ENV_PATHS[name], paths_raw.get(name, default))
        return _resolve_path(value, base)

    paths = PathsConfig(
        image_root=configured_path("image_root", "data/images"),
        starmap_root=configured_path("starmap_root", "data/starmaps"),
        glow_root=configured_path("glow_root", "data/glow"),
        output_root=configured_path("output_root", "outputs"),
        trajectory_root=configured_path("trajectory_root", "data/trajectories"),
    )
    run = raw.get("gneiss", {})
    trajectories = raw.get("trajectories", {})
    trajectory_397 = trajectories.get("397")
    trajectory_398 = trajectories.get("398")
    return GneissConfig(
        paths=paths,
        date=str(run.get("date", "20260210")),
        start=str(run.get("start", "10:19:00")),
        end=str(run.get("end", "10:28:40")),
        step_seconds=float(run.get("step_seconds", 1.0)),
        diagnostic_time=str(run.get("diagnostic_time", "10:25:20")),
        apex_reference_time=str(run.get("apex_reference_time", "10:24:00")),
        target_grid_spacing_km=float(run.get("target_grid_spacing_km", 2.0)),
        decimation=int(run.get("decimation", 1)),
        trajectory_397=_resolve_path(trajectory_397, base) if trajectory_397 else None,
        trajectory_398=_resolve_path(trajectory_398, base) if trajectory_398 else None,
    )


def validate_data_paths(config: GneissConfig) -> list[str]:
    """Return human-readable problems without requiring every optional dataset."""
    problems = []
    for name in ("image_root", "starmap_root", "glow_root"):
        path = getattr(config.paths, name)
        if not path.exists():
            problems.append(f"{name} does not exist: {path}")
    return problems
