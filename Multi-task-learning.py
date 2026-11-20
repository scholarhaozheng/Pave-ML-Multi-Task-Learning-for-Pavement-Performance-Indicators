"""Three-stage multi-task learning for pavement performance indicators.

Run with --help for dataset and experiment options.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import joblib
import keras_tuner as kt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tcn import TCN

from pave_ml.data import prepare_datasets


TARGET_NAMES = ("IRI", "RD", "PCI")


class MultiStream:
    """Write training logs to multiple streams."""

    def __init__(self, streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def safe_filename(value):
    return re.sub(r'[\\/:*?"<>|]', "_", value)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def plot_loss(history, title, output_dir):
    """Plot training and validation loss."""
    for log_scale in (False, True):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history["loss"], label="Training", color="#2878B5")
        ax.plot(history.history["val_loss"], label="Validation", color="#A5668B")
        ax.set(xlabel="Epoch", ylabel="Mean squared error loss", title=title)
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        suffix = "_log" if log_scale else ""
        fig.savefig(Path(output_dir) / f"{safe_filename(title)}{suffix}.pdf")
        plt.close(fig)


def convert_to_float(observation):
    value = getattr(observation, "value", observation)
    return float(np.asarray(value).reshape(-1)[0])


def plot_hyperparameter_search_best_epoch(tuner, title, output_dir):
    """Plot only observations belonging to the corresponding trial."""
    records = []
    for index, trial in enumerate(
        sorted(tuner.oracle.trials.values(), key=lambda item: item.trial_id), 1
    ):
        if trial.status != "COMPLETED":
            continue
        try:
            observations = trial.metrics.get_history("val_loss")
        except ValueError:
            continue
        if not observations:
            continue
        best = min(observations, key=convert_to_float)
        training = trial.metrics.get_history("loss")
        same_step = [item for item in training if item.step == best.step]
        records.append((index, convert_to_float(best),
                        convert_to_float(same_step[0]) if same_step else np.nan))
    if not records:
        return
    values = np.asarray(records)
    for log_scale in (False, True):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(values[:, 0], values[:, 1], "o-", label="Validation", color="#A5668B")
        ax.plot(values[:, 0], values[:, 2], "s-", label="Training at selected epoch",
                color="#2878B5")
        best = values[np.argmin(values[:, 1])]
        ax.scatter(best[0], best[1], color="#F39237", s=85,
                   label="Best validation loss", zorder=3)
        ax.set(xlabel="Trial", ylabel="Loss", title=title)
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        suffix = "_log" if log_scale else ""
        fig.savefig(Path(output_dir) / f"{safe_filename(title)}{suffix}.pdf")
        plt.close(fig)


def evaluate(true_pred, true_y, target_name, output_dir, sample_weight=None):
    """Report test metrics in original units, using observed targets only."""
    predicted = np.asarray(true_pred).reshape(-1)
    observed = np.asarray(true_y).reshape(-1)
    if predicted.shape != observed.shape:
        raise ValueError("Prediction and target lengths differ.")
    valid = np.isfinite(observed)
    if sample_weight is not None:
        weights = np.asarray(sample_weight).reshape(-1)
        if weights.shape != observed.shape:
            raise ValueError("Target mask and target lengths differ.")
        valid &= weights > 0
    if not np.all(np.isfinite(predicted[valid])):
        raise ValueError(f"{target_name}: non-finite predictions for observed targets.")
    predicted, observed = predicted[valid], observed[valid]
    if not len(observed):
        return {"n_observed": 0, "rmse": None, "mae": None,
                "mape_pct": None, "mape_n_nonzero": 0, "r2": None}
    nonzero = observed != 0
    report = {
        "n_observed": int(len(observed)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(observed, predicted))),
        "mae": float(metrics.mean_absolute_error(observed, predicted)),
        # Percentage error is undefined at zero: report its denominator explicitly.
        "mape_pct": float(np.mean(np.abs(
            (observed[nonzero] - predicted[nonzero]) / observed[nonzero])) * 100)
            if np.any(nonzero) else None,
        "mape_n_nonzero": int(np.sum(nonzero)),
        "r2": float(metrics.r2_score(observed, predicted))
            if len(observed) > 1 and np.ptp(observed) > 0 else None,
    }
    print(f"{target_name} — held-out test: {json.dumps(report, ensure_ascii=False)}")
    residuals = predicted - observed
    prefix = safe_filename(target_name)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=min(50, max(5, len(residuals) // 2)), color="#2878B5")
    ax.set(xlabel="Prediction − observation", ylabel="Count",
           title=f"{target_name}: test residuals")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"{prefix}_residual_hist.pdf")
    plt.close(fig)
    lower = min(observed.min(), predicted.min())
    upper = max(observed.max(), predicted.max())
    margin = max((upper - lower) * 0.08, 1e-6)
    bounds = (lower - margin, upper + margin)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(observed, predicted, s=10, alpha=0.5, color="#2878B5")
    axes[0].plot(bounds, bounds, color="#F39237")
    axes[0].set(xlabel="Observed", ylabel="Predicted", xlim=bounds, ylim=bounds,
                title=f"{target_name}: held-out test")
    axes[1].scatter(predicted, residuals, s=10, alpha=0.5, color="#17806D")
    axes[1].axhline(0, color="#F39237")
    axes[1].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs prediction")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"{prefix}_scatter.pdf")
    plt.close(fig)
    return report


@tf.keras.utils.register_keras_serializable(package="PaveML")
class TileContext(layers.Layer):
    """Repeat a TCN encoder context across the decoder's time dimension."""

    def call(self, inputs):
        context, decoder_input = inputs
        return tf.tile(tf.expand_dims(context, axis=1),
                       [1, tf.shape(decoder_input)[1], 1])

    def compute_output_shape(self, input_shape):
        context_shape, decoder_shape = input_shape
        return context_shape[0], decoder_shape[1], context_shape[-1]


class SimpleHP:
    """Read a fixed, previously selected branch configuration."""

    def __init__(self, values):
        self.values = dict(values)

    def Choice(self, key, values, default=None):
        return self.values.get(key, default)

    def Int(self, key, min_value, max_value, step=1, default=None):
        return self.values.get(key, default)

    def Boolean(self, key, default=False):
        return self.values.get(key, default)

    def Float(self, key, min_value, max_value, sampling="linear", default=None):
        return self.values.get(key, default)


def auto_map_hp(old_hp, old_branch, new_branch):
    old_prefix, new_prefix = f"task_{old_branch}_", f"task_{new_branch}_"
    return {
        new_prefix + key[len(old_prefix):]: value
        for key, value in old_hp.values.items() if key.startswith(old_prefix)
    }


def build_task_branch(shared_feature, branch_name, hp, decoder_input, frozen=False):
    """Conv1D feature extraction and an LSTM, GRU, or TCN encoder-decoder."""
    key = f"task_{branch_name}_"
    conv_activation = hp.Choice(key + "Conv1D_activation",
                                ["tanh", "relu", "selu", "elu"], default="tanh")
    output_activation = hp.Choice(key + "output_activation",
                                  ["tanh", "relu", "selu", "elu"], default="tanh")
    activation = hp.Choice(key + "activation",
                          ["tanh", "relu", "selu", "elu"], default="tanh")
    dense_activation = hp.Choice(key + "dense_activation",
                                ["tanh", "relu", "selu", "elu"], default="tanh")
    feature_units = hp.Choice(key + "feature_units", [10, 20, 40, 80], default=10)
    conv_filters = hp.Int(key + "conv_filters", 8, 64, step=8, default=16)
    kernel_size = hp.Choice(key + "kernel_size", [2, 3, 5, 8, 16], default=2)
    rnn_units = hp.Int(key + "rnn_units", 8, 64, step=8, default=16)
    num_encoder = hp.Int(key + "num_encoder_rnn", 1, 3, default=1)
    num_decoder = hp.Int(key + "num_decoder_rnn", 1, 3, default=1)
    rnn_type = hp.Choice(key + "rnn_type", ["LSTM", "GRU", "TCN"], default="GRU")
    fc_values = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    fc1 = hp.Choice(key + "fc1_units", fc_values, default=20)
    fc2 = hp.Choice(key + "fc2_units", fc_values, default=10)
    residual = hp.Boolean(key + "res_or_not", default=False)
    feature_dense = hp.Choice(key + "feature_dense", [32, 64, 128], default=64)
    training = False if frozen else None

    x = layers.BatchNormalization(name=f"{branch_name}_feature_bn")(
        shared_feature, training=training)
    x = layers.Dense(feature_dense, activation=dense_activation,
                     name=f"{branch_name}_feature_dense")(x)
    x = layers.Dropout(0.0 if frozen else 0.3,
                       name=f"{branch_name}_feature_dropout")(x, training=training)
    x = layers.Dense(feature_units, activation=dense_activation,
                     name=f"{branch_name}_feature_dense_2")(x)
    x = layers.Conv1D(conv_filters, kernel_size, padding="same",
                      activation=conv_activation, name=f"{branch_name}_conv")(x)
    x = layers.BatchNormalization(name=f"{branch_name}_bn")(x, training=training)

    if rnn_type in ("LSTM", "GRU"):
        cell = layers.LSTM if rnn_type == "LSTM" else layers.GRU
        for index in range(num_encoder):
            final = index == num_encoder - 1
            result = cell(
                rnn_units, return_sequences=True, return_state=final,
                dropout=0.0 if frozen else 0.2,
                recurrent_dropout=0.0 if frozen else 0.2,
                name=f"{branch_name}_encoder_{rnn_type.lower()}_{index + 1}",
            )(x, training=training)
            if final:
                x, *encoder_state = result
            else:
                x = result
        x = decoder_input
        for index in range(num_decoder):
            result = cell(
                rnn_units, return_sequences=True, return_state=True,
                dropout=0.0 if frozen else 0.2,
                recurrent_dropout=0.0 if frozen else 0.2,
                name=f"{branch_name}_decoder_{rnn_type.lower()}_{index + 1}",
            )(x, initial_state=encoder_state if index == 0 else None,
              training=training)
            x = result[0]
    else:
        tcn_filters = hp.Int(key + "tcn_filters", 8, 64, step=8, default=rnn_units)
        tcn_kernel = hp.Choice(key + "tcn_kernel_size", [2, 3, 5, 8, 16], default=2)
        for index in range(num_encoder):
            x = TCN(nb_filters=tcn_filters, kernel_size=tcn_kernel,
                    dilations=[1, 2, 4], padding="causal", return_sequences=True,
                    activation=activation,
                    name=f"{branch_name}_encoder_tcn_{index + 1}")(x, training=training)
        context = layers.GlobalAveragePooling1D(
            name=f"{branch_name}_encoder_context")(x)
        tiled = TileContext(name=f"{branch_name}_tile_context")([context, decoder_input])
        x = layers.Concatenate(name=f"{branch_name}_concat_decoder")([decoder_input, tiled])
        for index in range(num_decoder):
            x = TCN(nb_filters=tcn_filters, kernel_size=tcn_kernel,
                    dilations=[1, 2, 4], padding="causal", return_sequences=True,
                    activation=activation,
                    name=f"{branch_name}_decoder_tcn_{index + 1}")(x, training=training)

    if residual:
        pooled = layers.GlobalAveragePooling1D(name=f"{branch_name}_pool")(x)
        skip = layers.Conv1D(fc1, 1, padding="same", activation=activation,
                            name=f"{branch_name}_res_conv")(shared_feature)
        skip = layers.GlobalAveragePooling1D(name=f"{branch_name}_res_pool")(skip)
        projected = layers.Dense(fc1, activation=activation,
                                 name=f"{branch_name}_proj_x")(pooled)
        skip = layers.Dense(fc1, activation=activation,
                            name=f"{branch_name}_proj_res")(skip)
        x = layers.Add(name=f"{branch_name}_res_add")([projected, skip])
    else:
        x = layers.GlobalAveragePooling1D(name=f"{branch_name}_gap")(x)
    x = layers.Dense(fc1, activation=activation, name=f"{branch_name}_dense_1")(x)
    x = layers.Dropout(0.0 if frozen else 0.2,
                       name=f"{branch_name}_dropout_1")(x, training=training)
    x = layers.Dense(fc2, activation=activation, name=f"{branch_name}_dense_2")(x)
    x = layers.Dropout(0.0 if frozen else 0.2,
                       name=f"{branch_name}_dropout_2")(x, training=training)
    return layers.Dense(1, activation=output_activation, name=f"{branch_name}_output")(x)


def snapshot_weights(model):
    """Copy model weights into an independent snapshot."""
    return {
        layer.name: [np.array(weight, copy=True) for weight in layer.get_weights()]
        for layer in model.layers if layer.weights
    }


def restore_weights(model, snapshot, prefix_pairs):
    for layer in model.layers:
        if not layer.weights:
            continue
        for destination, source in prefix_pairs:
            if layer.name.startswith(destination):
                source_name = source + layer.name[len(destination):]
                if source_name not in snapshot:
                    raise ValueError(f"Missing transfer weights: {source_name}")
                layer.set_weights(snapshot[source_name])
                break


class StageHyperModel(kt.HyperModel):
    """Build each trial from the same selected earlier-stage weights."""

    def __init__(self, stage, feature_count, prior_hps=None, prior_weights=None):
        self.stage = stage
        self.feature_count = feature_count
        self.prior_hps = dict(prior_hps or {})
        self.prior_weights = prior_weights

    def build(self, hp):
        stage = self.stage
        shared_hp = hp if stage == "M1" else SimpleHP(self.prior_hps["M1"])
        if stage == "M1":
            hp.Choice("batch_size", [32, 64, 128], default=64)
        shared_activation = shared_hp.Choice(
            "shared_input_activation", ["relu", "selu", "tanh"], default="relu")
        input_units = shared_hp.Int("shared_input_units", 64, 256, step=64, default=128)
        feature_units = shared_hp.Int("shared_feature_units", 10, 210, step=50, default=10)
        encoder = layers.Input(shape=(None, self.feature_count + 3), name="encoder_input")
        decoder = layers.Input(shape=(1, 1), name="decoder_input")
        shared = layers.Dense(input_units, activation=shared_activation,
                              name="shared_feature_1")(encoder)
        shared = layers.Dropout(0.0 if stage == "M2" else 0.3,
                                name="shared_feature_dropout")(
            shared, training=False if stage == "M2" else None)
        shared = layers.Dense(feature_units, activation=shared_activation,
                              name="shared_feature")(shared)
        outputs = []
        if stage == "M1":
            outputs.append(build_task_branch(shared, "y1_M1", hp, decoder))
        else:
            iri_hp = SimpleHP(auto_map_hp(
                SimpleHP(self.prior_hps["M1"]), "y1_M1", f"y1_{stage}"))
            outputs.append(build_task_branch(
                shared, f"y1_{stage}", iri_hp, decoder, frozen=stage == "M2"))
            if stage == "M2":
                outputs.append(build_task_branch(shared, "y2_M2", hp, decoder))
            else:
                rd_hp = SimpleHP(auto_map_hp(
                    SimpleHP(self.prior_hps["M2"]), "y2_M2", "y2_M3"))
                outputs.append(build_task_branch(shared, "y2_M3", rd_hp, decoder))
                outputs.append(build_task_branch(shared, "y3_M3", hp, decoder))
        model = Model([encoder, decoder], outputs[0] if stage == "M1" else outputs,
                      name=stage)
        if stage == "M2":
            restore_weights(model, self.prior_weights,
                            [("shared_", "shared_"), ("y1_M2_", "y1_M1_")])
            for layer in model.layers:
                if layer.name.startswith(("shared_", "y1_M2_")):
                    layer.trainable = False
        elif stage == "M3":
            restore_weights(model, self.prior_weights,
                            [("shared_", "shared_"), ("y1_M3_", "y1_M2_"),
                             ("y2_M3_", "y2_M2_")])
        rate_key = "shared_learning_rate" if stage == "M1" else f"learning_rate_{stage}"
        rate = hp.Float(rate_key, 1e-6, 1e-1, sampling="log", default=1e-2)
        if stage == "M1":
            model.compile(tf.keras.optimizers.Adam(rate), loss="mean_squared_error")
        else:
            # M2 has no observed IRI labels. Its frozen output has zero loss weight.
            model.compile(tf.keras.optimizers.Adam(rate),
                          loss=["mean_squared_error"] * len(outputs),
                          loss_weights=[0.0, 1.0] if stage == "M2" else [1.0] * 3)
        return model

    def fit(self, hp, model, *args, **kwargs):
        if self.stage == "M1":
            kwargs["batch_size"] = hp.get("batch_size")
        return model.fit(*args, **kwargs)


def prepare_multi_task_labels(labels):
    targets, weights = [], []
    for label in labels:
        y = np.asarray(label, dtype=np.float32).reshape(-1, 1)
        mask = np.isfinite(y).reshape(-1).astype(np.float32)
        targets.append(np.where(np.isfinite(y), y, 0.0))
        weights.append(mask)
    return targets, weights


def model_data(partition, stage):
    targets, weights = prepare_multi_task_labels(partition["Y"])
    decoder = np.zeros((len(partition["X"]), 1, 1), dtype=np.float32)
    inputs = [partition["X"], decoder]
    if stage == "M1":
        return inputs, targets[0], weights[0]
    if stage == "M2":
        return inputs, [np.zeros_like(targets[0]), targets[0]], [
            np.zeros_like(weights[0]), weights[0]]
    return inputs, targets, weights


def fit_stage(stage, dataset, hypermodel, args, run_dir, batch_size):
    train_x, train_y, train_weights = model_data(dataset["train"], stage)
    val_x, val_y, val_weights = model_data(dataset["val"], stage)
    validation = (val_x, val_y, val_weights)
    tuner = kt.BayesianOptimization(
        hypermodel, objective="val_loss", max_trials=args.max_trials,
        executions_per_trial=args.executions_per_trial,
        directory=str(run_dir / "hyper_tuning"), project_name=stage,
        seed=args.seed, overwrite=False,
    )
    # Every run has a new directory, so trials from an older split are never reused.
    tuner.search(
        train_x, train_y, sample_weight=train_weights,
        epochs=args.tune_epochs, batch_size=batch_size, validation_data=validation,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience,
                                 restore_best_weights=True)],
        verbose=args.verbose,
    )
    selected_hp = tuner.get_best_hyperparameters(1)[0]
    selected_batch = selected_hp.get("batch_size") if stage == "M1" else batch_size
    write_json(run_dir / f"{stage}_hyperparameters.json", selected_hp.values)
    plot_hyperparameter_search_best_epoch(tuner, f"{stage} tuning", run_dir / "figures")
    model = hypermodel.build(selected_hp)
    model.summary()
    checkpoint = run_dir / "checkpoints" / f"{stage}.weights.h5"
    history = model.fit(
        train_x, train_y, sample_weight=train_weights,
        epochs=args.train_epochs, batch_size=selected_batch, validation_data=validation,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.patience,
                          restore_best_weights=True),
            ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True,
                            save_weights_only=True),
        ],
        verbose=args.verbose,
    )
    # Load the checkpoint selected by validation loss.
    model.load_weights(checkpoint)
    model.save(run_dir / "checkpoints" / f"{stage}.keras")
    write_json(run_dir / f"{stage}_history.json",
               {key: [float(value) for value in values]
                for key, values in history.history.items()})
    plot_loss(history, f"{stage} training", run_dir / "figures")
    return model, selected_hp, selected_batch


def run_experiment(args):
    tf.keras.utils.set_random_seed(args.seed)
    prepared = prepare_datasets(
        args.data_dir, seq_length=args.seq_length, seed=args.seed,
        group_map_path=args.group_map,
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("--run-name must be a single directory name.")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "figures").mkdir()
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(MultiStream([sys.stdout, log])):
            print(f"Run directory: {run_dir}")
            write_json(run_dir / "config.json", vars(args))
            write_json(run_dir / "split_manifest.json", prepared["manifest"])
            joblib.dump({
                "feature_scaler": prepared["feature_scaler"],
                "target_scalers": prepared["target_scalers"],
                "target_names": TARGET_NAMES,
                "seq_length": args.seq_length,
                "feature_count": prepared["feature_count"],
            }, run_dir / "preprocessing.joblib")
            models, selected_hps = {}, {}
            previous_weights = None
            batch_size = 64
            for stage in ("M1", "M2", "M3"):
                print(f"\n---------- {stage}: training and validation ----------")
                hypermodel = StageHyperModel(
                    stage, prepared["feature_count"], prior_hps=selected_hps,
                    prior_weights=previous_weights,
                )
                model, hp, batch_size = fit_stage(
                    stage, prepared["stages"][stage], hypermodel, args, run_dir, batch_size)
                models[stage] = model
                selected_hps[stage] = dict(hp.values)
                previous_weights = snapshot_weights(model)
            # All architectures and checkpoints are fixed before any test metrics.
            results = {}
            for stage, model in models.items():
                test = prepared["stages"][stage]["test"]
                test_x, _, _ = model_data(test, stage)
                predictions = model.predict(test_x, verbose=0)
                predictions = [predictions] if stage == "M1" else predictions
                tasks = [(0, 0, 0)] if stage == "M1" else (
                    [(1, 0, 1)] if stage == "M2" else [(i, i, i) for i in range(3)])
                results[stage] = {}
                for prediction_index, target_index, scaler_index in tasks:
                    scaler = prepared["target_scalers"][scaler_index]
                    prediction = scaler.inverse_transform(
                        np.asarray(predictions[prediction_index]).reshape(-1, 1))
                    # Keep exact observed zeros and original precision for metrics.
                    truth = np.asarray(test["Y_raw"][target_index]).reshape(-1, 1)
                    target_name = TARGET_NAMES[scaler_index]
                    results[stage][target_name] = evaluate(
                        prediction, truth, f"{stage}_{target_name}", run_dir / "figures")
            write_json(run_dir / "test_metrics.json", results)
            print(f"Saved test metrics and selected models in {run_dir}")
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="txt_csv", help="Directory of three numeric CSV files.")
    parser.add_argument("--group-map", required=True,
                        help="CSV mapping each source segment ID to a canonical road group.")
    parser.add_argument("--output-dir", default="runs", help="Parent directory for independent runs.")
    parser.add_argument("--run-name", help="New run subdirectory; an existing name is rejected.")
    parser.add_argument("--seq-length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--tune-epochs", type=int, default=30)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--max-trials", type=int, default=40)
    parser.add_argument("--executions-per-trial", type=int, default=3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args(argv)
    for key in ("seq_length", "tune_epochs", "train_epochs", "max_trials",
                "executions_per_trial"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.patience < 0:
        parser.error("--patience cannot be negative")
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
