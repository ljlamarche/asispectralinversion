"""Download a checksum-verified small example-data bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


DEFAULT_MANIFEST = (
    "https://raw.githubusercontent.com/317Lab/asispectralinversion/main/"
    "examples/example-data-manifest.json"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(location):
    if str(location).startswith(("http://", "https://")):
        with urllib.request.urlopen(str(location)) as response:
            return json.load(response)
    with Path(location).open(encoding="utf-8") as stream:
        return json.load(stream)


def download_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path, default=Path("data/example"))
    args = parser.parse_args(argv)
    manifest = read_json(args.manifest)
    args.destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        target = args.destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) == item["sha256"]:
            print(f"Verified {target}")
            continue
        urllib.request.urlretrieve(item["url"], target)
        actual = sha256(target)
        if actual != item["sha256"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {item['path']}")
        print(f"Downloaded {target}")
