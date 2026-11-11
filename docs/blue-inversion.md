# GNEISS RGB inversion

Open `gneiss_rgb.ipynb` from the repository root, or run:

```sh
gneiss-run --notebook gneiss_rgb.ipynb --executed-notebook outputs/gneiss-rgb-executed.ipynb
```

The notebook uses `config.toml` for times, paths, and grid settings. It estimates Q from 427.8 nm brightness using `calculate_E0_Q_v2`, then estimates E0 from red/green at that Q. All three channel footprints must cover a pixel for its site to be eligible for fixed ownership. Invalid owner values remain missing. The original `gneiss.ipynb` remains the RG comparison workflow.

Blue brightness is `(mean counts - background) / exposure_seconds * factor`, using factors ARV=100, VEE=40, BVR=10. These factors are stored in the shared calibration module under the exposure-normalized convention, R s/count. Exposure and frame cadence are separate quantities read from the acquisition logs. A three-frame mean is divided by one frame's exposure, not three times the exposure. No blue dark-frame subtraction is applied. Off-sky background estimation and wavelet denoising remain enabled.

The notebook uses the available blue starmaps:

- ARV: `blue/ARV/ZOR_ARV_20260210_4278_{Az,El}.FIT`
- BVR: `blue/BVR/ARI_20260209_4278_{Az,El}.FIT`
- VEE: `blue/VEE/VEE_4278_20260210_050100_{az,el}.sav`

Blue emission is mapped from 107 km to the common 110 km layer.

Blue stacks are discovered under `image_root/blue/<site>` with same-stem `.log` files. Logs supply Date, TimeStart, ImageCadence(s), Exposure(s), NumberOfImages, and Filter(Ang). The loader requires 4278 A and verifies the TIFF page count. The user verified that ARV THA observes 4278 A despite a stale 7774-A label. The notebook allows that exact mismatch for ARV only and records reported and verified wavelengths. BVR ARI and VEE must report 4278 A. Missing maps, other wavelength mismatches, and insufficient temporal coverage produce errors rather than an RG fallback.

Outputs use `gneiss_rgb_..._fixed_ownership.h5`; the file records blue input timing/exposure and calibration. Resuming with changed blue inputs or calibration is rejected. Outputs made with the earlier dark-frame subtraction also require a new output file or explicit recomputation with `overwrite_output=True`. A reduced-grid diagnostic check is distinct from validating a full mission inversion.

Only the fixed owner’s finite positive RGB samples are passed to Q/E0 and conductance inversion. Preprocessing still uses the full images. Per-site result maps contain NaNs outside owned pixels. `ownership_count` is not saved in RGB outputs; `owner_site` and `source_site` identify geometric ownership and valid contributions. Older outputs must be recomputed or written to a different file; they cannot be resumed into this format.

The RGB series processes timestamps in separate worker processes. Set `timestamp_workers = 2` under `[gneiss]` in `config.toml` (the default), or use `1` for sequential timestamps. Each worker processes its sites sequentially and reuses its prepared geometry, lookup tables, and frame cache. More workers require more memory and may compete for disk access; process startup can outweigh savings for short series. The inspect cell still processes a single timestamp.

Only the notebook process writes HDF5. Results can finish out of order but are stored at their original time indices and flushed after every completed timestamp. Set `overwrite_output = False` to resume unfinished timestamps. On Python 3.14+, interrupting or closing the series stops its workers; older Python versions wait for running tasks to finish. Restart the kernel and rerun setup after updating the worker code.

The merged map retains the fixed owner when either Q or E0 is finite; a missing component stays NaN and no other site fills it. `source_site` is nonzero for partial results. Outputs use the `fixed_geometric_owner_no_fallback_either_finite` policy identifier, so older outputs cannot be resumed into the changed policy; regenerate them or choose a new output path.

RGB outputs now include `combined/failure_reason`, a compressed uint8 cube with a versioned `codes_json` legend. Codes describe the first failed stage: input validity, blue lookup bounds/Q failure, E0 support/crossings/ambiguity/numerical failure, or conductance failure after finite Q/E0. Code 0 means success, 1 means outside ownership, and 255 is unrecorded. Codes are copied from the geometric owner even if both outputs are missing. The completion flag remains authoritative for interrupted writes.

`nan_analysis.ipynb` reads these saved codes with replay disabled by default. Older files cannot provide these reasons retroactively; use a new output file or explicitly regenerate with `overwrite_output=True`. Recording codes adds no second inversion or second ratio-curve search.
