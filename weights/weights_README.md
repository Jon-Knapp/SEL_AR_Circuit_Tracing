# Weights

## `device_detection/`: the delivered model and its training run

This folder is the complete training run that produced the detection model used by
the delivered system. The model itself is `device_detection/weights/best.pt`. The
copy that ships with the runnable code is `src/weights_v2_4_obb.pt` (the **same
model**), renamed so `config.py` can point at it. (`weights/last.pt`, if present, is
just the final-epoch checkpoint and is not needed to use the system.)

Everything else here is the training run's own output, kept as **provenance** so the
model's quality is transparent:

- `results.csv`, `results.png`: training metrics across epochs
- `BoxP_curve.png`, `BoxR_curve.png`, `BoxPR_curve.png`, `BoxF1_curve.png`:
  precision, recall, PR, and F1 curves
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `train_batch*.jpg`: sample training images
- `val_batch*_labels.jpg` / `val_batch*_pred.jpg`: validation images showing the
  ground-truth labels vs. the model's predictions
- `args.yaml`: the exact training settings used

The four detected classes are `Flathead_Block`, `Phillips_Block`, `Terminal_1`, and
`Terminal_2`.

## `probe_tracking/`: future work, NOT used by the delivered system

A trained model from the experimental markerless-probe-detection effort. The
delivered system does **not** use this: its probe tracking is color-based (HSV) and
uses no trained model (see `src/probe_tracking.py`). It is included only for whoever
continues that line of work. The dataset behind it is attached to the v1.0 Release
(see `datasets/probe_detection/README.md`).
