"""Command-line entry points for portable GNEISS workflows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import ENV_CONFIG, load_config, validate_data_paths


def config_main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect asispectralinversion configuration")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for name, value in vars(config.paths).items():
        print(f"{name}: {value}")
    if args.check:
        problems = validate_data_paths(config)
        if problems:
            parser.error("\n".join(problems))
        print("Required data roots exist.")


def gneiss_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Execute the configured GNEISS inversion notebook"
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--notebook", type=Path, default=Path("gneiss.ipynb"))
    parser.add_argument(
        "--executed-notebook", type=Path, default=Path("outputs/gneiss-executed.ipynb")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    problems = validate_data_paths(config)
    if problems:
        parser.error("\n".join(problems))
    print(f"Configuration: {config_path}")
    print(f"Images: {config.paths.image_root}")
    print(f"Starmaps: {config.paths.starmap_root}")
    print(f"GLOW: {config.paths.glow_root}")
    print(f"Range: {config.date} {config.start}–{config.end}, step={config.step_seconds:g}s")
    if args.dry_run:
        return

    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as error:
        parser.error(f"Notebook execution requires `pip install -e '.[gneiss]'`: {error}")

    notebook_path = args.notebook.expanduser().resolve()
    output_path = args.executed_notebook.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_CONFIG] = str(config_path)
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    nbformat.write(notebook, output_path)
    print(f"Wrote executed notebook: {output_path}")
