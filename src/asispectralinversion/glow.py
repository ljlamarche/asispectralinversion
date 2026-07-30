"""Generate portable GLOW input files from TOML configuration."""

import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .config import default_config_path, load_config


def generate_inputs(config_path=None, destination=None):
    config_path = Path(config_path or default_config_path()).resolve()
    config = load_config(config_path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    sites = raw.get("glow", {}).get("sites", {})
    if not sites:
        raise ValueError("No [glow.sites.*] entries are configured")
    destination = Path(destination or "src/glow_invert/inputs").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for site, values in sites.items():
        output = config.paths.glow_root / site
        output.mkdir(parents=True, exist_ok=True)
        line = (
            f"{values['yydoy']} {values['ut_seconds']} "
            f"{values['latitude']:.3f} {values['longitude_east']:.3f} "
            f"{values['f107a']:.1f} {values['f107']:.1f} "
            f"{values['f107_previous']:.1f} {values['ap']:.1f} "
            f"'{output}'\n"
        )
        path = destination / f"in.invert.{site.upper()}"
        path.write_text(line, encoding="utf-8")
        paths.append(path)
    return paths


def glow_input_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    try:
        paths = generate_inputs(args.config, args.destination)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)
