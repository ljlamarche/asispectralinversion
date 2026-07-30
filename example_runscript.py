#!/usr/bin/env python3
"""Run the legacy Poker Flat RGB API with externally stored example data."""

import argparse
import datetime as dt
from pathlib import Path

from asispectralinversion.filing import file_data, process_grouped_files
from asispectralinversion.transformation import feed_data


def legacy_image_groups(data_dir):
    """Return the three historical 2023-03-14 PKR frames per wavelength."""
    names = {
        "red": (
            "PKR_20230314_064904_0630.png",
            "PKR_20230314_064912_0630.png",
            "PKR_20230314_064920_0630.png",
        ),
        "green": (
            "PKR_20230314_064902_0558.png",
            "PKR_20230314_064910_0558.png",
            "PKR_20230314_064918_0558.png",
        ),
        "blue": (
            "PKR_20230314_064859_0428.png",
            "PKR_20230314_064907_0428.png",
            "PKR_20230314_064915_0428.png",
        ),
    }
    return {
        color: [str(data_dir / name) for name in filenames]
        for color, filenames in names.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--lookup-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-series", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    groups = legacy_image_groups(args.data_dir)
    missing = [
        path for paths in groups.values() for path in paths if not Path(path).exists()
    ]
    if missing:
        parser.error("missing legacy example image(s):\n" + "\n".join(missing))
    if not args.lookup_dir.exists():
        parser.error(f"lookup directory does not exist: {args.lookup_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    date = dt.date(2023, 3, 14)
    feed_data(
        date,
        groups["blue"],
        groups["green"],
        groups["red"],
        str(args.lookup_dir),
        str(args.output_dir / "test_out.hdf5"),
        plot=args.plot,
    )

    if not args.skip_series:
        timestamps, blue, green, red = file_data(
            date,
            dt.time(6, 48),
            dt.time(6, 52),
            str(args.data_dir),
        )
        process_grouped_files(
            timestamps,
            blue,
            green,
            red,
            str(args.lookup_dir),
            str(args.output_dir),
        )


if __name__ == "__main__":
    main()
