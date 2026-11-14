"""Reusable TIFF page counts; timestamps and cadence remain caller controlled."""

import json
import os
from pathlib import Path
import tempfile
import time
import warnings

import tifffile


class TiffPageCountCache:
    """Cache by absolute path, byte size, and nanosecond modification time.

    Atomic replacement keeps simultaneous jobs from reading partial JSON. Jobs
    may redundantly index a file or overwrite one another's new cache entries;
    a missing entry only causes a rescan, never a changed scientific result.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.entries = {}
        self.hits = 0
        self.misses = 0
        try:
            data = json.loads(self.path.read_text())
            if data.get('version') == 1 and isinstance(data.get('files'), dict):
                self.entries = data['files']
        except FileNotFoundError:
            pass
        except (OSError, ValueError, AttributeError) as exc:
            warnings.warn(f'Cannot read TIFF catalog {self.path}; rebuilding: {exc}')

    def page_count(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        signature = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        entry = self.entries.get(str(path))
        if (isinstance(entry, dict)
                and all(entry.get(key) == value for key, value in signature.items())
                and type(entry.get('pages')) is int and entry['pages'] > 0):
            self.hits += 1
            return entry['pages']

        print(f'Indexing TIFF: {path}', flush=True)
        started = time.monotonic()
        with tifffile.TiffFile(path) as tif:
            count = len(tif.pages)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f'TIFF changed while indexing: {path}; retry after acquisition finishes')
        self.entries[str(path)] = {**signature, 'pages': count}
        self.misses += 1
        self._save()
        print(f'  {count:,} pages in {time.monotonic() - started:.1f}s (catalog updated)', flush=True)
        return count

    def _save(self):
        # Checkpoint each new count so interrupted indexing can resume.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent,
                                             prefix=self.path.name + '.', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({'version': 1, 'files': self.entries}, stream, indent=2)
                stream.write('\n')
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
