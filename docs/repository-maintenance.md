# Repository data and history policy

## What belongs in Git

- Python and Fortran source
- Build files
- Configuration templates
- Documentation
- Small deterministic fixtures
- Dataset manifests and checksums

## What does not belong in ordinary Git

- Compiled `.exe`, `.o`, `.mod`, or platform libraries
- Full TIFF stacks and starmaps
- Generated GLOW lookup tables
- Inversion HDF5/NPZ outputs
- Executed notebooks containing large image outputs

Use a versioned institutional archive, Zenodo, GitHub release assets, Git LFS,
or DVC for scientific datasets. Record checksums and provenance in a manifest.

## Existing history

Removing files from the index prevents them from appearing in future commits,
but old clones still contain the historical blobs. This repository previously
had an approximately 950 MB packed Git history.

Rewriting shared history is intentionally not automated. If all collaborators
agree, make a backup and use `git filter-repo` to remove the historical
generated-data directories, then force-push during a coordinated migration.
Every collaborator must make a fresh clone afterward. Do not perform this
operation casually on an active shared repository.
