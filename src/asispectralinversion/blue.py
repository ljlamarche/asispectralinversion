"""Blue acquisition timing and exposure from camera sidecar logs."""
import datetime as dt
from pathlib import Path

import numpy as np
import tifffile


def blue_stack_catalog(directory, date, *, verified_stale_filter=None):
    """Catalog verified 4278-A stacks on a UTC date."""
    target_date = dt.datetime.strptime(date, '%Y%m%d').date()
    catalog = []
    for path in sorted(Path(directory).glob('*.tif*')):
        fields = dict(line.split(maxsplit=1) for line in
                      path.with_suffix('.log').read_text().splitlines()
                      if len(line.split(maxsplit=1)) == 2)
        start = dt.datetime.fromisoformat(fields['Date'] + 'T' + fields['TimeStart'])
        if start.date() != target_date:
            continue
        reported_filter = float(fields['Filter(Ang)'])
        if reported_filter != 4278 and reported_filter != verified_stale_filter:
            raise ValueError(f"{path}: log reports {fields['Filter(Ang)']} A, expected 4278 A; verify acquisition wavelength")
        exposure = float(fields['Exposure(s)'])
        cadence = float(fields['ImageCadence(s)'])
        if not np.isfinite([exposure, cadence]).all() or min(exposure, cadence) <= 0:
            raise ValueError(f'{path}: exposure and cadence must be positive and finite')
        with tifffile.TiffFile(path) as tif:
            count = len(tif.pages)
        if count != int(fields['NumberOfImages']):
            raise ValueError(f'{path}: TIFF page count disagrees with acquisition log')
        catalog.append(dict(path=str(path), start=start, count=count,
                            cadence=cadence, exposure=exposure,
                            reported_filter_ang=reported_filter,
                            verified_filter_ang=4278))
    if not catalog:
        raise FileNotFoundError(f'No blue stacks for {date} in {directory}')
    return catalog


def coadd_blue_frames(catalog, target, count, read_frame):
    """Average nearest frames, rejecting gaps and mixed exposure settings."""
    if count < 1:
        raise ValueError('Frame count must be positive')
    candidates = []
    for stack in catalog:
        cadence = stack['cadence']
        center = round((target - stack['start']).total_seconds() / cadence)
        for index in range(max(0, center-count), min(stack['count'], center+count+1)):
            timestamp = stack['start'] + dt.timedelta(seconds=index*cadence)
            offset = abs((timestamp-target).total_seconds())
            if offset <= count*cadence:
                candidates.append((offset, timestamp, stack['path'], index, stack['exposure'], cadence))
    selected = sorted(candidates)[:count]
    if len(selected) != count or selected[0][0] > selected[0][5]:
        raise ValueError(f'Insufficient blue coverage near {target.isoformat()} UTC')
    exposures = {item[4] for item in selected}
    if len(exposures) != 1:
        raise ValueError('Selected blue frames have differing exposures')
    selected.sort(key=lambda item: item[1])
    return (np.mean([read_frame(item[2], item[3]) for item in selected], axis=0),
            [Path(item[2]).name for item in selected], [item[3] for item in selected],
            [item[1] for item in selected], exposures.pop())
