"""Behavioral checks for model transfer, masking, and held-out evaluation.

These checks use small random tensors and require the training dependencies.
"""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

try:
    import tensorflow as tf
    import keras_tuner as kt
    from tcn import TCN
except ModuleNotFoundError as error:
    raise unittest.SkipTest("Training dependencies are not installed: " + error.name)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("pave_ml_training", ROOT / "Multi-task-learning.py")
training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training
SPEC.loader.exec_module(training)


def small_hyperparameters(branch, architecture="GRU", depth=1, residual=False):
    hp = kt.HyperParameters()
    hp.Fixed("shared_input_units", 64)
    hp.Fixed("shared_feature_units", 10)
    hp.Fixed("shared_learning_rate", 0.001)
    hp.Fixed("learning_rate_M2", 0.001)
    hp.Fixed("learning_rate_M3", 0.001)
    hp.Fixed("batch_size", 32)
    prefix = f"task_{branch}_"
    settings = {
        "rnn_type": architecture, "rnn_units": 8, "conv_filters": 8,
        "kernel_size": 2, "feature_units": 10, "feature_dense": 32,
        "num_encoder_rnn": depth, "num_decoder_rnn": depth,
        "fc1_units": 5, "fc2_units": 5,
        "res_or_not": residual,
        "tcn_filters": 8, "tcn_kernel_size": 2,
    }
    for name, value in settings.items():
        hp.Fixed(prefix + name, value)
    return hp


def random_inputs(count=6):
    random = np.random.default_rng(19)
    return [random.normal(size=(count, 3, 5)).astype("float32"),
            np.zeros((count, 1, 1), dtype="float32")]


class ModelBehaviorTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(37)

    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_all_encoder_families_round_trip_saved_predictions(self):
        inputs = random_inputs()
        with tempfile.TemporaryDirectory() as directory:
            for architecture in ("LSTM", "GRU", "TCN"):
                with self.subTest(architecture=architecture):
                    hp = small_hyperparameters("y1_M1", architecture, residual=True)
                    model = training.StageHyperModel("M1", 2).build(hp)
                    expected = np.asarray(model(inputs, training=False))
                    self.assertEqual(expected.shape, (6, 1))
                    self.assertTrue(np.isfinite(expected).all())
                    path = Path(directory) / f"{architecture}.keras"
                    model.save(path)
                    restored = tf.keras.models.load_model(
                        path, custom_objects={"TCN": TCN}, compile=False)
                    actual = np.asarray(restored(inputs, training=False))
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_frozen_stage_and_trial_initialization_are_independent(self):
        inputs = random_inputs()
        labels = np.linspace(-0.4, 0.6, 6, dtype="float32").reshape(-1, 1)
        hp1 = small_hyperparameters("y1_M1", depth=2)
        m1 = training.StageHyperModel("M1", 2).build(hp1)
        m1.train_on_batch(inputs, labels)
        snapshot1 = training.snapshot_weights(m1)
        untouched1 = {name: [value.copy() for value in values]
                      for name, values in snapshot1.items()}
        hp2 = small_hyperparameters("y2_M2", depth=2)
        m2 = training.StageHyperModel(
            "M2", 2, {"M1": hp1.values}, snapshot1).build(hp2)
        frozen = {layer.name: [value.copy() for value in layer.get_weights()]
                  for layer in m2.layers
                  if layer.weights and layer.name.startswith(("shared_", "y1_M2_"))}
        shared = tf.keras.Model(m2.inputs[0], m2.get_layer("shared_feature").output)
        # A frozen feature extractor must not apply training-time dropout.
        first = np.asarray(shared(inputs[0], training=True))
        second = np.asarray(shared(inputs[0], training=True))
        np.testing.assert_array_equal(first, second)
        before_rd = m2.get_layer("y2_M2_output").get_weights()[0].copy()
        m2.train_on_batch(
            inputs, [np.zeros_like(labels), labels],
            sample_weight=[np.zeros(6, dtype="float32"), np.ones(6, dtype="float32")])
        for name, values in frozen.items():
            for expected, actual in zip(values, m2.get_layer(name).get_weights()):
                np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(before_rd, m2.get_layer("y2_M2_output").get_weights()[0]))
        for name, values in untouched1.items():
            for expected, actual in zip(values, snapshot1[name]):
                np.testing.assert_array_equal(actual, expected)

        snapshot2 = training.snapshot_weights(m2)
        untouched2 = {name: [value.copy() for value in values]
                      for name, values in snapshot2.items()}
        hp3 = small_hyperparameters("y3_M3")
        factory = training.StageHyperModel(
            "M3", 2, {"M1": hp1.values, "M2": hp2.values}, snapshot2)
        first_trial = factory.build(hp3)
        first_trial.train_on_batch(inputs, [labels, labels, labels])
        second_trial = factory.build(hp3)
        for prefix, source in (("shared_", "shared_"),
                               ("y1_M3_", "y1_M2_"), ("y2_M3_", "y2_M2_")):
            for layer in second_trial.layers:
                if layer.weights and layer.name.startswith(prefix):
                    source_name = source + layer.name[len(prefix):]
                    for expected, actual in zip(untouched2[source_name], layer.get_weights()):
                        np.testing.assert_array_equal(actual, expected)
        for name, values in untouched2.items():
            for expected, actual in zip(values, snapshot2[name]):
                np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(
            first_trial.get_layer("shared_feature_1").get_weights()[0],
            second_trial.get_layer("shared_feature_1").get_weights()[0]))

    def test_unobserved_targets_cannot_change_weighted_loss(self):
        inputs = random_inputs()
        hp1 = small_hyperparameters("y1_M1")
        m1 = training.StageHyperModel("M1", 2).build(hp1)
        hp2 = small_hyperparameters("y2_M2")
        m2 = training.StageHyperModel(
            "M2", 2, {"M1": hp1.values}, training.snapshot_weights(m1)).build(hp2)
        m3 = training.StageHyperModel(
            "M3", 2, {"M1": hp1.values, "M2": hp2.values},
            training.snapshot_weights(m2)).build(small_hyperparameters("y3_M3"))
        labels = [np.zeros((6, 1), dtype="float32") for _ in range(3)]
        weights = [np.ones(6, dtype="float32") for _ in range(3)]
        weights[0][::2] = 0
        weights[2][1::2] = 0
        m3.reset_metrics()
        baseline = m3.test_on_batch(inputs, labels, sample_weight=weights, return_dict=True)
        changed = [value.copy() for value in labels]
        changed[0][::2] = 10000
        changed[2][1::2] = -10000
        m3.reset_metrics()
        masked = m3.test_on_batch(inputs, changed, sample_weight=weights, return_dict=True)
        self.assertAlmostEqual(baseline["loss"], masked["loss"], places=6)
        changed[1][:] = 10000
        m3.reset_metrics()
        observed = m3.test_on_batch(inputs, changed, sample_weight=weights, return_dict=True)
        self.assertGreater(observed["loss"], baseline["loss"] + 1000)


class TrainingProtocolTests(unittest.TestCase):
    def test_search_and_final_fit_receive_masks_and_validation_only(self):
        partition = {"X": random_inputs(3)[0],
                     "Y": [np.array([[1.0], [np.nan], [3.0]], dtype="float32"),
                           np.array([[np.nan], [2.0], [3.0]], dtype="float32"),
                           np.array([[1.0], [2.0], [np.nan]], dtype="float32")]}

        class TrainingPartitions(dict):
            def __getitem__(self, key):
                if key == "test":
                    raise AssertionError("Model selection attempted to read test data.")
                return super().__getitem__(key)

        dataset = TrainingPartitions(train=partition, val=partition)
        args = SimpleNamespace(max_trials=1, executions_per_trial=1, seed=2,
                               tune_epochs=1, train_epochs=1, patience=0, verbose=0)
        expected_by_stage = {
            "M1": [np.array([1, 0, 1])],
            "M2": [np.array([0, 0, 0]), np.array([1, 0, 1])],
            "M3": [np.array([1, 0, 1]), np.array([0, 1, 1]), np.array([1, 1, 0])],
        }
        for stage, expected in expected_by_stage.items():
            with self.subTest(stage=stage):
                fake_model = Mock()
                fake_model.fit.return_value = SimpleNamespace(
                    history={"loss": [1.0], "val_loss": [2.0]})
                fake_hp = SimpleNamespace(values={}, get=lambda key: 32)
                fake_tuner = Mock()
                fake_tuner.get_best_hyperparameters.return_value = [fake_hp]
                hypermodel = Mock()
                hypermodel.build.return_value = fake_model
                with tempfile.TemporaryDirectory() as directory:
                    with patch.object(training.kt, "BayesianOptimization", return_value=fake_tuner), \
                            patch.object(training, "plot_loss"), \
                            patch.object(training, "plot_hyperparameter_search_best_epoch"):
                        training.fit_stage(stage, dataset, hypermodel, args, Path(directory), 32)
                for call in (fake_tuner.search.call_args, fake_model.fit.call_args):
                    train_weights = call.kwargs["sample_weight"]
                    val_weights = call.kwargs["validation_data"][2]
                    if stage == "M1":
                        train_weights, val_weights = [train_weights], [val_weights]
                    for actual, wanted in zip(train_weights, expected):
                        np.testing.assert_array_equal(actual, wanted)
                    for actual, wanted in zip(val_weights, expected):
                        np.testing.assert_array_equal(actual, wanted)
                    for target in call.args[1]:
                        self.assertTrue(np.isfinite(target).all())
                    self.assertTrue(all(callback.monitor == "val_loss"
                                        for callback in call.kwargs["callbacks"]))

    def test_selected_batch_size_reaches_the_search_fit(self):
        hp = small_hyperparameters("y1_M1")
        fake_model = Mock()
        training.StageHyperModel("M1", 2).fit(hp, fake_model, [], [], batch_size=999)
        self.assertEqual(fake_model.fit.call_args.kwargs["batch_size"], 32)

    def test_all_models_are_selected_before_test_predictions(self):
        from sklearn.preprocessing import StandardScaler
        events = []
        prepared = {"feature_count": 2, "manifest": {},
                    "feature_scaler": StandardScaler().fit([[0, 0], [1, 1]]),
                    "target_scalers": [StandardScaler().fit([[0], [1]]) for _ in range(3)],
                    "stages": {}}
        for stage, count in (("M1", 1), ("M2", 1), ("M3", 3)):
            prepared["stages"][stage] = {
                split: {"X": np.full((3, 3, 5), marker, dtype="float32"),
                        "Y": [np.ones((3, 1), dtype="float32") for _ in range(count)],
                        "Y_raw": [np.array([[0.0], [1.0], [2.0]]) for _ in range(count)]}
                for split, marker in (("train", 1), ("val", 2), ("test", 999))
            }

        def select(stage, dataset, hypermodel, args, run_dir, batch_size):
            events.append("fit_" + stage)
            model = Mock()
            model.layers = []

            def predict(inputs, **kwargs):
                self.assertEqual(events[:3], ["fit_M1", "fit_M2", "fit_M3"])
                self.assertTrue(np.all(inputs[0] == 999))
                events.append("predict_" + stage)
                result = np.zeros((3, 1))
                return result if stage == "M1" else [result] * (2 if stage == "M2" else 3)

            model.predict.side_effect = predict
            return model, SimpleNamespace(values={}), 32

        with tempfile.TemporaryDirectory() as directory:
            args = training.parse_args(["--group-map", "unused.csv",
                                        "--output-dir", directory, "--run-name", "test"])
            with patch.object(training, "prepare_datasets", return_value=prepared), \
                    patch.object(training, "fit_stage", side_effect=select), \
                    patch.object(training, "evaluate", return_value={"n_observed": 3}) as evaluate:
                run = training.run_experiment(args)
            self.assertTrue((run / "test_metrics.json").exists())
            self.assertEqual(evaluate.call_count, 5)
            for call in evaluate.call_args_list:
                np.testing.assert_array_equal(call.args[1], [[0.0], [1.0], [2.0]])
            self.assertEqual(events, ["fit_M1", "fit_M2", "fit_M3",
                                      "predict_M1", "predict_M2", "predict_M3"])

    def test_evaluation_flattens_masks_and_checks_observed_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            report = training.evaluate(
                np.array([[2.0], [99.0], [5.0]]), np.array([[1.0], [2.0], [3.0]]),
                "synthetic", directory, sample_weight=np.array([[1], [0], [1]]))
            self.assertEqual(report["n_observed"], 2)
            self.assertAlmostEqual(report["rmse"], np.sqrt(2.5))
            self.assertAlmostEqual(report["mae"], 1.5)
            with self.assertRaisesRegex(ValueError, "non-finite predictions"):
                training.evaluate([np.nan], [1.0], "bad", directory)
            empty = training.evaluate([np.nan], [np.nan], "missing", directory)
            self.assertEqual(empty["n_observed"], 0)
            self.assertIsNone(empty["rmse"])

    def test_mape_excludes_exact_observed_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            report = training.evaluate([1.0, 2.0, 4.0], [0.0, 1.0, 2.0],
                                       "zero_reference", directory)
        self.assertEqual(report["n_observed"], 3)
        self.assertEqual(report["mape_n_nonzero"], 2)
        self.assertEqual(report["mape_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
