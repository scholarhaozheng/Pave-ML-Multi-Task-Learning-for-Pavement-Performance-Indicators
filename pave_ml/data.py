"""Prepare datasets with physical-section splits and training-fitted scalers.

Input files are headerless CSV text. Each single-task row contains a segment
identifier, F features and its target; a multi-task row contains an identifier,
the same F features and IRI, RD, PCI targets. Rows within each segment must be
chronological. Distinct identifiers are required across gaps in a segment's
history. The mapping CSV links these source-specific segment identifiers to
physical road groups so that transfer-learning stages share one split.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


FILENAMES = {
    "iri": "cleaned_result_array_IRI.txt",
    "rd": "cleaned_result_array_RD.txt",
    "multi": "cleaned_result_array_IRI_RDI_with_PCI.txt",
}
HISTORY_MISSING_VALUE = 100.0
SPLITS = ("train", "val", "test")


def _read_numeric_file(path, target_count, feature_count=None):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or all(not value.strip() for value in row):
                continue
            row = [value.strip() for value in row]
            if not row[0]:
                raise ValueError(f"{path.name}:{line_number}: segment ID is empty")
            rows.append((line_number, row))
    if not rows:
        raise ValueError(f"{path.name}: the data file is empty")
    if feature_count is None:
        feature_count = len(rows[0][1]) - 1 - target_count
    if feature_count < 1:
        raise ValueError(f"{path.name}: at least one input feature is required")
    expected = 1 + feature_count + target_count
    identifiers, values = [], []
    for line_number, row in rows:
        if len(row) != expected:
            raise ValueError(
                f"{path.name}:{line_number}: expected {expected} columns "
                f"(ID, {feature_count} features, {target_count} target(s)); got {len(row)}"
            )
        try:
            numeric = [float(value) if value else np.nan for value in row[1:]]
        except ValueError as error:
            raise ValueError(
                f"{path.name}:{line_number}: features and targets must be numeric; "
                "files have no header"
            ) from error
        identifiers.append(row[0])
        values.append(numeric)
    values = np.asarray(values, dtype=np.float64)
    features = values[:, :feature_count]
    targets = values[:, feature_count:]
    if not np.isfinite(features).all():
        raise ValueError(f"{path.name}: input features must be finite")
    if np.isinf(targets).any():
        raise ValueError(f"{path.name}: targets may be NaN, but not infinite")
    return {
        "ids": np.asarray(identifiers, dtype=str),
        "features": features,
        "targets": targets,
    }


def _read_group_map(path, datasets):
    if path is None:
        raise ValueError(
            "A group mapping CSV is required. Supply --group-map with columns "
            "source,id,group; source must be iri, rd or multi. Map segments from "
            "the same physical road to the same group across all three files."
        )
    path = Path(path)
    mappings = {source: {} for source in FILENAMES}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"source", "id", "group"}.issubset(reader.fieldnames):
            raise ValueError("The group mapping CSV must contain source,id,group headers")
        for line_number, row in enumerate(reader, 2):
            source, identifier, group = (str(row.get(key) or "").strip()
                                         for key in ("source", "id", "group"))
            if source not in mappings or not identifier or not group:
                raise ValueError(f"{path.name}:{line_number}: invalid source, id or group")
            if identifier in mappings[source]:
                raise ValueError(f"{path.name}:{line_number}: duplicate mapping for {source}/{identifier}")
            mappings[source][identifier] = group
    for source, dataset in datasets.items():
        identifiers = set(dataset["ids"])
        missing = identifiers - mappings[source].keys()
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ValueError(f"Group mapping is incomplete for {source}: {preview}")
        unknown = mappings[source].keys() - identifiers
        if unknown:
            preview = ", ".join(sorted(unknown)[:5])
            raise ValueError(f"Group mapping contains unknown {source} IDs: {preview}")
        dataset["groups"] = np.asarray([mappings[source][identifier]
                                        for identifier in dataset["ids"]], dtype=str)
    return mappings


def _split_groups(datasets, seed):
    groups = sorted({group for data in datasets.values() for group in data["groups"]})
    try:
        train_val, test = train_test_split(groups, test_size=0.15, random_state=seed)
        train, val = train_test_split(train_val, test_size=15 / 85, random_state=seed)
    except ValueError as error:
        raise ValueError("Not enough distinct road groups for a train/validation/test split") from error
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def _target_scaler(values, name):
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError(f"No observed training targets are available to fit the {name} scaler")
    return StandardScaler().fit(finite.reshape(-1, 1))


def _scaled_targets(values, scalers):
    transformed = np.full(values.shape, np.nan, dtype=np.float64)
    for task, scaler in enumerate(scalers):
        observed = np.isfinite(values[:, task])
        if observed.any():
            transformed[observed, task] = scaler.transform(
                values[observed, task].reshape(-1, 1)
            ).ravel()
    return transformed


def _windows(data, source, group_split, feature_scaler, target_scalers, seq_length):
    task_indices = {"iri": [0], "rd": [1], "multi": [0, 1, 2]}[source]
    scaled_features = feature_scaler.transform(data["features"])
    scaled_targets = _scaled_targets(data["targets"], [target_scalers[i] for i in task_indices])
    history = np.full((len(data["ids"]), 3), HISTORY_MISSING_VALUE, dtype=np.float64)
    for local_task, global_task in enumerate(task_indices):
        observed = np.isfinite(scaled_targets[:, local_task])
        history[observed, global_task] = scaled_targets[observed, local_task]
    inputs = np.concatenate((scaled_features, history), axis=1)
    results = {}
    for split in SPLITS:
        allowed_groups = set(group_split[split])
        X, Y, Y_raw, groups, segment_ids, target_rows = [], [], [], [], [], []
        for identifier in dict.fromkeys(data["ids"]):
            row_indices = np.flatnonzero(data["ids"] == identifier)
            group = data["groups"][row_indices[0]]
            if group not in allowed_groups:
                continue
            for start in range(len(row_indices) - seq_length):
                history_rows = row_indices[start:start + seq_length]
                target_row = row_indices[start + seq_length]
                target = scaled_targets[target_row]
                # An entirely unlabelled future row supplies no supervised loss.
                if not np.isfinite(target).any():
                    continue
                X.append(inputs[history_rows])
                Y.append(target)
                Y_raw.append(data["targets"][target_row].copy())
                groups.append(group)
                segment_ids.append(identifier)
                target_rows.append(int(target_row))
        if not X:
            raise ValueError(
                f"{source}/{split}: no labelled windows; each split needs road "
                f"segments with more than {seq_length} chronological rows"
            )
        y_values = np.asarray(Y, dtype=np.float32)
        raw_targets = np.asarray(Y_raw, dtype=np.float64)
        x_values = np.asarray(X, dtype=np.float32)
        if not np.isfinite(x_values).all() or np.isinf(y_values).any():
            raise ValueError(f"{source}/{split}: scaled values exceed the finite float32 range")
        if split == "train":
            for local_task, global_task in enumerate(task_indices):
                if not np.isfinite(y_values[:, local_task]).any():
                    name = ("IRI", "RD", "PCI")[global_task]
                    raise ValueError(f"{source}/train: no observed {name} targets in prediction windows")
        results[split] = {
            "X": x_values,
            "Y": [y_values[:, i:i + 1] for i in range(len(task_indices))],
            "Y_raw": [raw_targets[:, i:i + 1] for i in range(len(task_indices))],
            "groups": np.asarray(groups, dtype=str),
            "segment_ids": np.asarray(segment_ids, dtype=str),
            "target_rows": np.asarray(target_rows, dtype=np.int64),
        }
    return results


def prepare_datasets(data_dir, seq_length=3, seed=2022, group_map_path=None):
    """Split raw road groups, fit training-only scalers and build history windows.

    Returns all stages with train/val/test entries. Each entry has an X array of
    shape (samples, seq_length, F + 3), a list of scaled (samples, 1) Y arrays,
    and matching float64 Y_raw arrays for evaluation in the original units.
    Keeping raw labels preserves exact zeros for percentage-error metrics. Target
    NaNs remain available for loss masks. All model inputs are finite; unavailable
    IRI/RD/PCI history slots use the fixed value HISTORY_MISSING_VALUE.

    A feature scaler and IRI scaler fitted on M1 training rows are reused across
    stages. The RD scaler is fitted on M2 training rows and the PCI scaler on M3
    training rows. This preserves the transferred branches' scaling conventions.
    The returned scalers and JSON-compatible manifest can be saved by the caller.
    """
    if not isinstance(seq_length, int) or isinstance(seq_length, bool) or seq_length < 1:
        raise ValueError("seq_length must be a positive integer")
    if group_map_path is None:
        _read_group_map(None, {})
    data_dir = Path(data_dir)
    datasets = {"iri": _read_numeric_file(data_dir / FILENAMES["iri"], 1)}
    feature_count = datasets["iri"]["features"].shape[1]
    datasets["rd"] = _read_numeric_file(data_dir / FILENAMES["rd"], 1, feature_count)
    datasets["multi"] = _read_numeric_file(data_dir / FILENAMES["multi"], 3, feature_count)
    mappings = _read_group_map(group_map_path, datasets)
    group_split = _split_groups(datasets, seed)
    training_masks = {source: np.isin(data["groups"], group_split["train"])
                      for source, data in datasets.items()}
    if not training_masks["iri"].any():
        raise ValueError("No IRI training rows are available to fit the feature scaler")
    feature_scaler = MinMaxScaler().fit(datasets["iri"]["features"][training_masks["iri"]])
    target_scalers = [
        _target_scaler(datasets["iri"]["targets"][training_masks["iri"], 0], "IRI"),
        _target_scaler(datasets["rd"]["targets"][training_masks["rd"], 0], "RD"),
        _target_scaler(datasets["multi"]["targets"][training_masks["multi"], 2], "PCI"),
    ]
    stages = {stage: _windows(datasets[source], source, group_split, feature_scaler,
                             target_scalers, seq_length)
              for stage, source in (("M1", "iri"), ("M2", "rd"), ("M3", "multi"))}
    manifest = {
        "protocol": "physical-road-group-holdout",
        "seed": int(seed),
        "requested_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "seq_length": seq_length,
        "feature_count": feature_count,
        "history_order": ["IRI", "RD", "PCI"],
        "history_missing_value": HISTORY_MISSING_VALUE,
        "groups": group_split,
        "group_counts": {split: len(groups) for split, groups in group_split.items()},
        "scaler_fit_sources": {"features": "M1/train", "IRI": "M1/train",
                               "RD": "M2/train", "PCI": "M3/train"},
        "source_files": {source: {"filename": filename,
                                   "sha256": hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()}
                         for source, filename in FILENAMES.items()},
        "group_map_sha256": hashlib.sha256(Path(group_map_path).read_bytes()).hexdigest(),
        "segment_groups": mappings,
        "stages": {
            stage: {
                split: {
                    "samples": int(len(data["X"])),
                    "groups": sorted(set(data["groups"])),
                    "observed_targets": [int(np.isfinite(y).sum()) for y in data["Y"]],
                }
                for split, data in partitions.items()
            }
            for stage, partitions in stages.items()
        },
    }
    return {"stages": stages, "feature_count": feature_count,
            "feature_scaler": feature_scaler, "target_scalers": target_scalers,
            "manifest": manifest}
