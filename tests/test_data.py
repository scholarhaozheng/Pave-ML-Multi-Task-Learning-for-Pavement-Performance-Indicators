"""Regression checks for group separation and training-only preprocessing."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pave_ml.data import FILENAMES, HISTORY_MISSING_VALUE, prepare_datasets


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.rows = {source: [] for source in FILENAMES}
        self.mapping = []
        for road in range(20):
            for source in FILENAMES:
                identifier = f"{source}-segment-{road:02}"
                self.mapping.append([source, identifier, f"road-{road:02}"])
                for step in range(6):
                    features = [10 * road + step, 1000 + road]
                    iri, rd, pci = 1.0 + road + step / 10, 20.0 + road + step, 90.0 - road - step
                    targets = {"iri": [iri], "rd": [rd], "multi": [iri, rd, pci]}[source]
                    self.rows[source].append([identifier, *features, *targets])
        self.write_files()

    def tearDown(self):
        self.temporary.cleanup()

    def write_files(self):
        for source, filename in FILENAMES.items():
            with (self.directory / filename).open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(self.rows[source])
        self.map_path = self.directory / "groups.csv"
        with self.map_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source", "id", "group"])
            writer.writerows(self.mapping)

    def prepare(self, **kwargs):
        return prepare_datasets(self.directory, group_map_path=self.map_path, **kwargs)

    def test_cross_source_physical_roads_are_disjoint(self):
        result = self.prepare()
        groups_by_split = result["manifest"]["groups"]
        self.assertEqual(result["manifest"]["group_counts"], {"train": 14, "val": 3, "test": 3})
        self.assertFalse(set(groups_by_split["train"]) & set(groups_by_split["val"]))
        self.assertFalse(set(groups_by_split["train"]) & set(groups_by_split["test"]))
        self.assertFalse(set(groups_by_split["val"]) & set(groups_by_split["test"]))
        for partitions in result["stages"].values():
            for split, dataset in partitions.items():
                self.assertEqual(set(dataset["groups"]), set(groups_by_split[split]))
                self.assertEqual(len(dataset["X"]), 3 * len(groups_by_split[split]))

    def test_held_out_extremes_do_not_change_any_fitted_scaler_or_training_array(self):
        before = self.prepare()
        held_out = set(before["manifest"]["groups"]["val"] + before["manifest"]["groups"]["test"])
        mapped = {(source, identifier): group for source, identifier, group in self.mapping}
        for source, rows in self.rows.items():
            for row in rows:
                if mapped[(source, row[0])] in held_out:
                    row[1:] = [value + 1_000_000 for value in row[1:]]
        self.write_files()
        after = self.prepare()
        np.testing.assert_array_equal(before["feature_scaler"].data_min_, after["feature_scaler"].data_min_)
        np.testing.assert_array_equal(before["feature_scaler"].data_max_, after["feature_scaler"].data_max_)
        for left, right in zip(before["target_scalers"], after["target_scalers"]):
            np.testing.assert_array_equal(left.mean_, right.mean_)
            np.testing.assert_array_equal(left.var_, right.var_)
        for stage in before["stages"]:
            np.testing.assert_array_equal(before["stages"][stage]["train"]["X"],
                                          after["stages"][stage]["train"]["X"])
            for left, right in zip(before["stages"][stage]["train"]["Y"],
                                   after["stages"][stage]["train"]["Y"]):
                np.testing.assert_array_equal(left, right)
        self.assertGreater(after["stages"]["M1"]["test"]["X"][:, :, 0].max(), 1.0)

    def test_scaler_statistics_match_training_raw_rows(self):
        result = self.prepare()
        train_groups = set(result["manifest"]["groups"]["train"])
        expected_rows = {}
        for source in FILENAMES:
            mapping = {identifier: group for src, identifier, group in self.mapping if src == source}
            expected_rows[source] = np.asarray([row[1:] for row in self.rows[source]
                                                if mapping[row[0]] in train_groups], dtype=float)
        np.testing.assert_array_equal(result["feature_scaler"].data_min_, expected_rows["iri"][:, :2].min(axis=0))
        np.testing.assert_array_equal(result["feature_scaler"].data_max_, expected_rows["iri"][:, :2].max(axis=0))
        for index, values in enumerate([expected_rows["iri"][:, -1], expected_rows["rd"][:, -1],
                                        expected_rows["multi"][:, -1]]):
            self.assertAlmostEqual(result["target_scalers"][index].mean_[0], values.mean())

    def test_ids_are_not_scaled_or_fed_to_the_model(self):
        result = self.prepare()
        self.assertEqual(result["feature_count"], 2)
        self.assertEqual(result["feature_scaler"].n_features_in_, 2)
        for stage in result["stages"].values():
            for dataset in stage.values():
                self.assertEqual(dataset["X"].shape[1:], (3, 5))
                self.assertTrue(np.isfinite(dataset["X"]).all())

    def test_rd_history_uses_only_the_rd_slot(self):
        result = self.prepare()
        rd = result["stages"]["M2"]["train"]
        np.testing.assert_array_equal(rd["X"][:, :, 2], HISTORY_MISSING_VALUE)
        np.testing.assert_array_equal(rd["X"][:, :, 4], HISTORY_MISSING_VALUE)
        first_target_row = rd["target_rows"][0]
        expected_history = np.asarray([self.rows["rd"][row][-1]
                                       for row in range(first_target_row - 3, first_target_row)]).reshape(-1, 1)
        expected = result["target_scalers"][1].transform(expected_history).ravel()
        np.testing.assert_allclose(rd["X"][0, :, 3], expected, rtol=1e-6)

    def test_targets_exclude_the_future_from_the_input(self):
        result = self.prepare()
        for stage, source in (("M1", "iri"), ("M2", "rd"), ("M3", "multi")):
            for dataset in result["stages"][stage].values():
                for sample, row_index in enumerate(dataset["target_rows"]):
                    raw_history = np.asarray([self.rows[source][row][1:3]
                                               for row in range(row_index - 3, row_index)])
                    expected = result["feature_scaler"].transform(raw_history)
                    np.testing.assert_allclose(dataset["X"][sample, :, :2], expected, atol=1e-7)
                    self.assertTrue(all(self.rows[source][row][0] == dataset["segment_ids"][sample]
                                        for row in range(row_index - 3, row_index + 1)))

    def test_gap_segments_stay_separate_even_with_the_same_group(self):
        for source in FILENAMES:
            old_id = f"{source}-segment-00"
            new_id = f"{source}-segment-00-after-gap"
            self.mapping.append([source, new_id, "road-00"])
            for row in self.rows[source][3:6]:
                self.assertEqual(row[0], old_id)
                row[0] = new_id
        self.write_files()
        result = self.prepare()
        # Each of the two segments has only three rows, so neither can supply a
        # three-step history plus a future target.
        for stage in result["stages"].values():
            for dataset in stage.values():
                self.assertNotIn("road-00", dataset["groups"])

    def test_partial_missing_targets_remain_maskable_and_history_is_finite(self):
        for index, row in enumerate(self.rows["multi"]):
            if index % 6 in (1, 3):
                row[3] = "nan"  # IRI target/history slot.
        self.write_files()
        result = self.prepare()
        for dataset in result["stages"]["M3"].values():
            self.assertTrue(np.isnan(dataset["Y"][0]).any())
            self.assertTrue(np.isfinite(dataset["Y"][1]).all())
            self.assertTrue(np.isfinite(dataset["X"]).all())
            self.assertTrue((dataset["X"][:, :, 2] == HISTORY_MISSING_VALUE).any())

    def test_raw_labels_preserve_exact_zeros_for_percentage_error(self):
        values = [0.1, 0.2, 0.3, 0.0, 3.2, 0.4]
        for index, row in enumerate(self.rows["iri"]):
            row[-1] = values[index % len(values)]
        self.write_files()
        result = self.prepare()
        for dataset in result["stages"]["M1"].values():
            raw = dataset["Y_raw"][0]
            self.assertEqual(raw.dtype, np.float64)
            self.assertTrue((raw == 0).any())
            reconstructed = result["target_scalers"][0].inverse_transform(dataset["Y"][0])
            # A float32 training label can round-trip to a small nonzero value.
            # The retained raw label still identifies these samples as exact zero.
            self.assertTrue(np.any(reconstructed[raw == 0] != 0))
            np.testing.assert_array_equal(raw.ravel(), [values[int(row) % len(values)]
                                                       for row in dataset["target_rows"]])

    def test_raw_labels_keep_target_order_and_missingness_aligned_with_windows(self):
        for index, row in enumerate(self.rows["multi"]):
            if index % 6 == 4:
                row[3] = "nan"
            if index % 6 == 5:
                row[4] = "nan"
        self.write_files()
        result = self.prepare()
        for stage, source in (("M1", "iri"), ("M2", "rd"), ("M3", "multi")):
            for dataset in result["stages"][stage].values():
                expected = np.asarray([self.rows[source][int(row)][3:]
                                       for row in dataset["target_rows"]], dtype=np.float64)
                self.assertEqual(len(dataset["Y_raw"]), expected.shape[1])
                for task, (raw, scaled) in enumerate(zip(dataset["Y_raw"], dataset["Y"])):
                    self.assertEqual(raw.shape, scaled.shape)
                    np.testing.assert_array_equal(raw, expected[:, task:task + 1])
                    np.testing.assert_array_equal(np.isnan(raw), np.isnan(scaled))

    def test_empty_target_cells_are_treated_as_missing(self):
        for index, row in enumerate(self.rows["multi"]):
            if index % 6 == 3:
                row[-1] = ""
        self.write_files()
        result = self.prepare()
        self.assertTrue(np.isnan(result["stages"]["M3"]["train"]["Y"][2]).any())

    def test_mapping_is_mandatory_and_complete(self):
        with self.assertRaisesRegex(ValueError, "group mapping CSV is required"):
            prepare_datasets(self.directory)
        self.mapping.pop()
        self.write_files()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.prepare()

    def test_duplicate_mapping_is_rejected(self):
        self.mapping.append(self.mapping[0])
        self.write_files()
        with self.assertRaisesRegex(ValueError, "duplicate mapping"):
            self.prepare()

    def test_no_observed_training_target_is_rejected(self):
        before = self.prepare()
        train_groups = set(before["manifest"]["groups"]["train"])
        mapped = {identifier: group for source, identifier, group in self.mapping if source == "rd"}
        for row in self.rows["rd"]:
            if mapped[row[0]] in train_groups:
                row[-1] = "nan"
        self.write_files()
        with self.assertRaisesRegex(ValueError, "No observed training targets.*RD"):
            self.prepare()

    def test_history_only_labels_do_not_count_as_supervised_training_targets(self):
        before = self.prepare()
        train_groups = set(before["manifest"]["groups"]["train"])
        mapped = {identifier: group for source, identifier, group in self.mapping if source == "multi"}
        for index, row in enumerate(self.rows["multi"]):
            if mapped[row[0]] in train_groups and index % 6 >= 3:
                row[-1] = "nan"
        self.write_files()
        with self.assertRaisesRegex(ValueError, "multi/train: no observed PCI targets in prediction windows"):
            self.prepare()

    def test_manifest_is_json_serializable_and_seed_reproducible(self):
        first = self.prepare()
        second = self.prepare()
        self.assertEqual(first["manifest"], second["manifest"])
        json.dumps(first["manifest"], allow_nan=False)
        for stage in first["stages"]:
            for split in first["stages"][stage]:
                np.testing.assert_array_equal(first["stages"][stage][split]["X"],
                                              second["stages"][stage][split]["X"])


if __name__ == "__main__":
    unittest.main()
