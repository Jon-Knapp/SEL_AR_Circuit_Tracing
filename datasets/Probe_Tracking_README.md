# Probe-detection dataset (future work)

**This dataset is NOT used by the delivered system.** The shipped probe tracking is
color-based (HSV) and uses no trained model — see `src/probe_tracking.py`. This
dataset supports an experimental, future-work effort to track the probes with a
trained object detector instead.

## Why it isn't stored here

The dataset is roughly 476 MB zipped, which exceeds GitHub's 100 MB per-file limit
for files committed to a repository. It is therefore attached to the project's
GitHub **Release** rather than committed into this folder.

## Download

Get it from the v1.0 release:

https://github.com/Jon-Knapp/SEL_AR_Circuit_Tracing/releases/tag/v1.0

Look for the probe-detection dataset asset (a ~476 MB `.zip`).

## Related

A trained model produced from this dataset is in `weights/probe_tracking/`. Like the
dataset, it is future-work and is not part of the delivered system.
