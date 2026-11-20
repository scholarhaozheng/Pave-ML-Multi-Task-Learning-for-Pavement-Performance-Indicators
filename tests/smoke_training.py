"""Test all three training stages on synthetic road records.

Usage: python tests/smoke_training.py [--output-dir PATH]
The default uses a temporary directory and removes its artifacts on success.
An explicit output directory retains the synthetic inputs and model outputs.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pave_ml.data import FILENAMES


def make_synthetic_files(directory):
    """Make matching physical groups with deliberately different source IDs."""
    directory.mkdir(parents=True, exist_ok=True)
    mapping = []
    for source, filename in FILENAMES.items():
        rows = []
        for road in range(20):
            identifier = f"synthetic-{source}-{road:02}"
            mapping.append((source, identifier, f"synthetic-road-{road:02}"))
            for step in range(5):
                features = [road / 20 + step / 50, step / 10]
                iri, rd, pci = 1.0 + road / 50 + step / 20, 2.0 + road / 30 + step / 10, 90 - road / 3 - step
                targets = {"iri": [iri], "rd": [rd], "multi": [iri, rd, pci]}[source]
                if source == "multi" and road % 2 == 0 and step == 3:
                    targets[0] = "nan"
                rows.append((identifier, *features, *targets))
        with (directory / filename).open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
    group_map = directory / "group_map.csv"
    with group_map.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source", "id", "group"))
        writer.writerows(mapping)
    return group_map


def run_smoke(directory):
    spec = importlib.util.spec_from_file_location("pave_ml_training", ROOT / "Multi-task-learning.py")
    training = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = training
    spec.loader.exec_module(training)
    directory = Path(directory)
    mapping = make_synthetic_files(directory / "synthetic_data")
    args = training.parse_args([
        "--data-dir", str(directory / "synthetic_data"), "--group-map", str(mapping),
        "--output-dir", str(directory / "runs"), "--tune-epochs", "1",
        "--train-epochs", "1", "--max-trials", "1", "--executions-per-trial", "1",
        "--patience", "0", "--verbose", "0",
    ])
    run = training.run_experiment(args)
    results = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    assert set(results) == {"M1", "M2", "M3"}
    assert set(results["M1"]) == {"IRI"}
    assert set(results["M2"]) == {"RD"}
    assert set(results["M3"]) == {"IRI", "RD", "PCI"}
    for stage in results:
        assert (run / "checkpoints" / f"{stage}.keras").is_file()
        assert (run / "checkpoints" / f"{stage}.weights.h5").is_file()
        for report in results[stage].values():
            assert report["n_observed"] > 0 and report["rmse"] is not None
    print(f"Synthetic three-stage smoke test passed: {run}")
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Retain synthetic files and run artifacts here.")
    options = parser.parse_args()
    if options.output_dir:
        Path(options.output_dir).mkdir(parents=True, exist_ok=False)
        run_smoke(Path(options.output_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="pave-ml-smoke-") as temporary:
            run_smoke(Path(temporary))
