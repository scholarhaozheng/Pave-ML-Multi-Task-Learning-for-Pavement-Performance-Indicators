import copy
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# from tensorflow.keras.utils import plot_model
from contextlib import redirect_stdout
from tensorflow.keras.layers import Input, Concatenate
from keras_self_attention import SeqSelfAttention
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTMCell, GRUCell, RNN, Bidirectional
from tensorflow.keras import activations
from tcn import TCN

import matplotlib.pyplot as plt
import os
class MultiStream:
    def __init__(self, streams):
        self.streams = streams
    def write(self, message):
        for stream in self.streams:
            stream.write(message)
    def flush(self):
        for stream in self.streams:
            stream.flush()

log_file_path = os.path.join("txt_csv", "run_output_test.txt")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
log_file = open(log_file_path, "w", encoding="utf-8")
sys.stdout = MultiStream([sys.__stdout__, log_file])
print("所有 Evaluation 输出及其它日志均将同时显示在控制台并写入到:", log_file_path)

output_folder = "results_images_test"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print("图片将保存在文件夹：", output_folder)

epochs_num_hp = 30
epochs_num_train = 100
max_trials_num = 40
executions_per_trial_num = 3
num_trials_num = 5
patience_num = 7

input_seq_len = 3
n_out_features = 1
predict_labels = 3

def safe_filename(s):
    return re.sub(r'[\\/:*?"<>|]', '_', s)

def plot_loss(history, title, filename=None):
    train_color = '#2878B5'
    test_color = '#C497B2'

    plt.figure(figsize=(12, 6))
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Train', color=train_color)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Test', color=test_color)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.title(title)
    plt.legend()

    if filename is None:
        safe_title = safe_filename(title.replace(" ", "_"))
        filename = f"{safe_title}.pdf"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    plt.close()
    print("损失图保存到：", filepath)

    safe_title = safe_filename(title.replace(" ", "_"))
    plt.figure(figsize=(12, 6))
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Train', color=train_color)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Test', color=test_color)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss (log scale)')
    plt.title(title + " - Log Scale")
    plt.yscale('log')
    plt.legend()

    log_filename = f"{safe_title}_log.pdf"
    log_filepath = os.path.join(output_folder, log_filename)
    plt.savefig(log_filepath)
    plt.close()
    print("对数损失图保存到：", log_filepath)

def evaluate(true_pred, true_y, target_name="Target", epsilon=1e-10, filename_prefix=None, sample_weight=None):
    if filename_prefix is None:
        filename_prefix = f"evaluation_{target_name}"
    filename_prefix = safe_filename(filename_prefix)
    print("true_pred NaN 数量:", np.sum(np.isnan(true_pred)))
    print("true_y NaN 数量:", np.sum(np.isnan(true_y)))
    if sample_weight is not None:
        valid_mask = sample_weight > 0
        if np.sum(valid_mask) == 0:
            print("没有有效样本用于评估。")
            return
        eval_true_pred = true_pred[valid_mask]
        eval_true_y = true_y[valid_mask]
    else:
        eval_true_pred = true_pred
        eval_true_y = true_y
    print("eval_true_pred NaN 数量:", np.sum(np.isnan(eval_true_pred)))
    print("eval_true_y NaN 数量:", np.sum(np.isnan(eval_true_y)))
    plt.figure(figsize=(10, 6))
    residuals = eval_true_pred - eval_true_y
    plt.hist(residuals, bins=500)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    hist_filename = os.path.join(output_folder, f"{filename_prefix}_residual_hist.pdf")
    plt.savefig(hist_filename)
    plt.close()
    print("残差直方图保存到：", hist_filename)
    rmse = np.sqrt(metrics.mean_squared_error(eval_true_y, eval_true_pred))
    mae = metrics.mean_absolute_error(eval_true_y, eval_true_pred)
    mape = np.mean(np.abs((eval_true_y - eval_true_pred) / np.where(eval_true_y == 0, epsilon, eval_true_y))) * 100
    r2 = metrics.r2_score(eval_true_y, eval_true_pred)
    print("RMSE: %.4f" % rmse)
    print("MAE: %.4f" % mae)
    print("MAPE: %.4f" % mape)
    print("R2: %.4f" % r2)
    x_min, x_max = np.min(eval_true_y), np.max(eval_true_y)
    y_min, y_max = np.min(eval_true_pred), np.max(eval_true_pred)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    x_lim_lower = x_min - x_margin
    x_lim_upper = x_max + x_margin
    y_lim_lower = y_min - y_margin
    y_lim_upper = y_max + y_margin

    common_lower = min(x_lim_lower, y_lim_lower)
    common_upper = max(x_lim_upper, y_lim_upper)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(eval_true_y, eval_true_pred, s=1, c='b', alpha=0.5)
    ax1.set_xlabel("True {} Value".format(target_name))
    ax1.set_ylabel("Predicted {} Value".format(target_name))
    ax1.plot([common_lower, common_upper], [common_lower, common_upper], c='r')
    ax1.set_xlim(common_lower, common_upper)
    ax1.set_ylim(common_lower, common_upper)
    ax1.set_title("True vs Predicted {}".format(target_name), fontsize=12)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(eval_true_pred, residuals, s=1, c='g', alpha=0.5)
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_xlabel("Predicted {} Value".format(target_name))
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuals vs Predicted", fontsize=12)

    scatter_filename = os.path.join(output_folder, f"{filename_prefix}_scatter.pdf")
    plt.savefig(scatter_filename)
    plt.close()
    print("散点图和残差图保存到：", scatter_filename)


def convert_to_float(obs):
    x = obs.value if hasattr(obs, 'value') else obs
    if isinstance(x, list):
        return float(x[0])
    return float(x)

def safe_get_history(trial, metric_name):
    try:
        return trial.metrics.get_history(metric_name)
    except Exception as e:
        print(f"无法获取 {metric_name}：{e}")
        return None

def plot_hyperparameter_search_best_epoch(
    tuner,
    title,
    filename,
    train_color='#82B0D2',
    test_color='#FFBE7A',
    best_color='#FA7F6F'
):
    trials = tuner.oracle.trials
    trial_list = list(trials.values())
    trial_list.sort(key=lambda t: t.trial_id)

    trial_indices = []
    best_val_losses = []
    best_train_losses = []

    for idx, trial in enumerate(trial_list):
        val_history = safe_get_history(trial, "val_loss")
        train_history = safe_get_history(trial, "loss")
        if not (val_history and train_history):
            if idx + 1 < len(trial_list):
                next_trial = trial_list[idx + 1]
                print(f"Trial {trial.trial_id} 缺少 'val_loss' 或 'loss'，使用 Trial {next_trial.trial_id} 的指标")
                val_history = safe_get_history(next_trial, "val_loss")
                train_history = safe_get_history(next_trial, "loss")
            else:
                continue

        if val_history and train_history:
            val_loss_vals = [convert_to_float(obs) for obs in val_history]
            train_loss_vals = [convert_to_float(obs) for obs in train_history]
            best_val = min(val_loss_vals)
            best_train = min(train_loss_vals)
            trial_indices.append(idx + 1)
            best_val_losses.append(best_val)
            best_train_losses.append(best_train)

    if not trial_indices:
        print(f"{title} 没有可用的 trial 数据。")
        return

    overall_best_val = min(best_val_losses)
    overall_best_trial_index = trial_indices[best_val_losses.index(overall_best_val)]

    plt.figure(figsize=(10, 6))
    plt.plot(trial_indices, best_val_losses, 'o-', color=test_color, label="test")
    plt.plot(trial_indices, best_train_losses, 's-', color=train_color, label="train")
    plt.scatter([overall_best_trial_index], [overall_best_val], color=best_color,
                s=100, label="Overall Best (test)", zorder=10)
    plt.xlabel("Trial Index")
    plt.ylabel("Loss")
    plt.title(title + " (Linear Scale)")
    plt.legend()
    plt.grid(True)

    linear_file_path = os.path.join(output_folder, filename)
    plt.savefig(linear_file_path)
    plt.close()
    print(f"{title} (Linear Scale) 图像已保存到: {linear_file_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(trial_indices, best_val_losses, 'o-', color=test_color, label="test")
    plt.plot(trial_indices, best_train_losses, 's-', color=train_color, label="train")
    plt.scatter([overall_best_trial_index], [overall_best_val], color=best_color,
                s=100, label="Overall Best (test)", zorder=10)
    plt.xlabel("Trial Index")
    plt.ylabel("Loss")
    plt.title(title + " (Log Scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both')

    base, ext = os.path.splitext(filename)
    log_file_path = os.path.join(output_folder, base + "_log" + ext)
    plt.savefig(log_file_path)
    plt.close()
    print(f"{title} (Log Scale) 图像已保存到: {log_file_path}")

def build_task_branch(shared_feature, branch_name, n_out_features,
                      hp=None, prefix="",
                      default_conv_filters=16, default_kernel_size=2, default_rnn_units=16,
                      default_fc_units=[20, 10], default_activation=activations.tanh,
                      decoder_input=None, use_cudnn=False):
    if hp is not None:
        Conv1D_activation = hp.Choice(prefix + branch_name + '_Conv1D_activation',
                                      values=['tanh', 'relu', 'selu', 'elu'], default='tanh')
        output_activation = hp.Choice(prefix + branch_name + '_output_activation',
                                      values=['tanh', 'relu', 'selu', 'elu'], default='tanh')
        activation = hp.Choice(prefix + branch_name + '_activation', values=['tanh', 'relu', 'selu', 'elu'],
                               default='tanh')
        Dense_activation = hp.Choice(prefix + branch_name + '_dense_activation', values=['tanh', 'relu', 'selu', 'elu'],
                               default='tanh')
        feature_units = hp.Choice(prefix + branch_name + "_feature_units", values=[10, 20, 40, 80],default=10)
        conv_filters = hp.Int(prefix + branch_name + '_conv_filters', min_value=8, max_value=64, step=8,
                              default=default_conv_filters)
        kernel_size = hp.Choice(prefix + branch_name + '_kernel_size', values=[2, 3, 5, 8, 16],
                                default=default_kernel_size)
        rnn_units = hp.Int(prefix + branch_name + '_rnn_units', min_value=8, max_value=64, step=8,
                           default=default_rnn_units)
        num_encoder_rnn = hp.Int(prefix + branch_name + "num_encoder_rnn", min_value=1, max_value=3, step=1, default=1)
        num_decoder_rnn = hp.Int(prefix + branch_name + "num_decoder_rnn", min_value=1, max_value=3, step=1, default=1)
        rnn_type = hp.Choice(prefix + branch_name + '_rnn_type', values=['LSTM', 'GRU', 'TCN'], default='GRU')
        fc1_units = hp.Int(prefix + branch_name + '_fc1_units', min_value=5, max_value=45, step=10,
                           default=default_fc_units[0])
        fc2_units = hp.Int(prefix + branch_name + '_fc2_units', min_value=5, max_value=45, step=10,
                           default=default_fc_units[1])
        res_or_not = hp.Boolean(prefix + branch_name + '_res_or_not', default=False)
        num_feature_dense = hp.Choice(prefix + branch_name + '_feature_dense', values=[32, 64, 128], default=64)
        if rnn_type == 'TCN':
            tcn_filters = hp.Int(prefix + branch_name + '_tcn_filters', min_value=8, max_value=64, step=8,
                                 default=rnn_units)
            tcn_kernel_size = hp.Choice(prefix + branch_name + '_tcn_kernel_size', values=[2, 3, 5, 8, 16],
                                        default=default_kernel_size)
            tcn_dilations = [1, 2, 4]
    else:
        Conv1D_activation = default_activation
        activation = default_activation
        conv_filters = default_conv_filters
        kernel_size = default_kernel_size
        rnn_units = default_rnn_units
        num_encoder_rnn = 1
        num_decoder_rnn = 1
        rnn_type = 'GRU'
        fc1_units, fc2_units = default_fc_units
        res_or_not = False

    x = shared_feature
    normalized_feature = tf.keras.layers.BatchNormalization(name=f'{branch_name}_feature_bn')(x)
    feature_dense = Dense(num_feature_dense, activation=Dense_activation, name=f'{branch_name}_feature_dense')(normalized_feature)
    x = tf.keras.layers.Dropout(0.3, name=f'{branch_name}_feature_dropout')(feature_dense)
    x = Dense(feature_units, activation=Dense_activation, name=f'{branch_name}_feature_dense_2')(x)
    x = tf.keras.layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, padding='same',
                               activation=Conv1D_activation, name=f'{branch_name}_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{branch_name}_bn')(x)

    dropout_value = 0.0 if use_cudnn else 0.2
    recurrent_dropout_value = 0.0 if use_cudnn else 0.2

    if rnn_type in ['LSTM', 'GRU']:
        for i in range(num_encoder_rnn):
            return_state = (i == num_encoder_rnn - 1)
            if rnn_type == 'LSTM':
                lstm_layer = tf.keras.layers.LSTM(
                    rnn_units, return_sequences=True, return_state=return_state,
                    dropout=dropout_value, recurrent_dropout=recurrent_dropout_value, name=f'{branch_name}_encoder_lstm_{i+1}'
                )
                if return_state:
                    x, h, c = lstm_layer(x)
                    encoder_state = [h, c]
                else:
                    x = lstm_layer(x)
            else:
                gru_layer = tf.keras.layers.GRU(
                    rnn_units, return_sequences=True, return_state=return_state,
                    dropout=dropout_value, recurrent_dropout=recurrent_dropout_value, name=f'{branch_name}_encoder_gru_{i+1}'
                )
                if return_state:
                    x, state = gru_layer(x)
                    encoder_state = state
                else:
                    x = gru_layer(x)
    elif rnn_type == 'TCN':
        for i in range(num_encoder_rnn):
            x = TCN(
                nb_filters=tcn_filters, kernel_size=tcn_kernel_size, dilations=tcn_dilations,
                padding='causal', return_sequences=True, activation=activation,
                name=f'{branch_name}_encoder_tcn_{i+1}'
            )(x)
        encoder_context = tf.keras.layers.GlobalAveragePooling1D(name=f'{branch_name}_encoder_context')(x)
        encoder_state = encoder_context
    else:
        raise ValueError("Unsupported rnn_type: {}".format(rnn_type))

    if decoder_input is None:
        decoder_input = Input(shape=(None, n_out_features), name='decoder_input')
    x_dec = decoder_input
    if rnn_type in ['LSTM', 'GRU']:
        for i in range(num_decoder_rnn):
            if rnn_type == 'LSTM':
                lstm_layer = tf.keras.layers.LSTM(
                    rnn_units, return_sequences=True, return_state=True,
                    dropout=dropout_value, recurrent_dropout=recurrent_dropout_value, name=f'{branch_name}_decoder_lstm_{i+1}'
                )
                if i == 0 and encoder_state is not None:
                    x_dec, h, c = lstm_layer(x_dec, initial_state=encoder_state)
                else:
                    x_dec, h, c = lstm_layer(x_dec)
            else:
                gru_layer = tf.keras.layers.GRU(
                    rnn_units, return_sequences=True, return_state=True,
                    dropout=dropout_value, recurrent_dropout=recurrent_dropout_value, name=f'{branch_name}_decoder_gru_{i+1}'
                )
                if i == 0 and encoder_state is not None:
                    x_dec, state = gru_layer(x_dec, initial_state=encoder_state)
                else:
                    x_dec, state = gru_layer(x_dec)
        x = x_dec
    elif rnn_type == 'TCN':
        def tile_context(inputs):
            context, dec_in = inputs
            time_steps = tf.shape(dec_in)[1]
            context_expanded = tf.expand_dims(context, axis=1)
            return tf.tile(context_expanded, [1, time_steps, 1])
        context_tiled = tf.keras.layers.Lambda(tile_context, name=f'{branch_name}_tile_context')([encoder_state, decoder_input])
        x_dec = tf.keras.layers.Concatenate(name=f'{branch_name}_concat_decoder')([decoder_input, context_tiled])
        for i in range(num_decoder_rnn):
            x_dec = TCN(
                nb_filters=tcn_filters, kernel_size=tcn_kernel_size, dilations=tcn_dilations,
                padding='causal', return_sequences=True, activation=activation,
                name=f'{branch_name}_decoder_tcn_{i+1}'
            )(x_dec)
        x = x_dec
    else:
        raise ValueError("Unsupported rnn_type: {}".format(rnn_type))

    if res_or_not:
        pooled_x = tf.keras.layers.GlobalAveragePooling1D(name=f'{branch_name}_pool')(x)
        res = tf.keras.layers.Conv1D(
            filters=fc1_units, kernel_size=1, padding='same',
            activation=activation, name=f'{branch_name}_res_conv'
        )(shared_feature)
        res = tf.keras.layers.GlobalAveragePooling1D(name=f'{branch_name}_res_pool')(res)
        x_proj = tf.keras.layers.Dense(fc1_units, activation=activation, name=f'{branch_name}_proj_x')(pooled_x)
        res_proj = tf.keras.layers.Dense(fc1_units, activation=activation, name=f'{branch_name}_proj_res')(res)
        x = tf.keras.layers.Add(name=f'{branch_name}_res_add')([x_proj, res_proj])
    else:
        if x.shape[1] is None:
            x = tf.keras.layers.GlobalAveragePooling1D(name=f'{branch_name}_gap')(x)
        else:
            x = tf.keras.layers.Flatten(name=f'{branch_name}_flatten')(x)

    x = tf.keras.layers.Dense(fc1_units, activation=activation, name=f'{branch_name}_dense_1')(x)
    x = tf.keras.layers.Dropout(0.2, name=f'{branch_name}_dropout_1')(x)
    x = tf.keras.layers.Dense(fc2_units, activation=activation, name=f'{branch_name}_dense_2')(x)
    x = tf.keras.layers.Dropout(0.2, name=f'{branch_name}_dropout_2')(x)
    output = tf.keras.layers.Dense(n_out_features, activation=output_activation, name=f'{branch_name}_output')(x)
    return output, decoder_input

def load_data_single(filepath):
    dat = np.loadtxt(filepath, delimiter=',')
    num = dat.shape[1]
    raw_x = dat[:, :num - 1]
    raw_y = dat[:, num - 1].reshape(-1, 1)
    return raw_x, raw_y

raw_x, raw_y = load_data_single(f"txt_csv{os.sep}cleaned_result_array_IRI.txt")
n_in_features = raw_x.shape[1] - 1

def create_dataset_single_with_y_history(x, y, seq_length, output_seq_len=1, n_out_features=1, IRI_or_RD = "IRI"):
    unique_ids = np.unique(x[:, 0])
    X, Y = [], []

    for seg_id in unique_ids:
        mask = (x[:, 0] == seg_id)
        temp_dat = x[mask]
        num_cols = temp_dat.shape[1]
        raw_y = y[mask].reshape(-1, 1)
        raw_x = temp_dat[:, 1:num_cols]
        size = raw_y.shape[0]

        for i in range(size - seq_length):
            x_seq = raw_x[i:(i + seq_length), :]
            y_seq = raw_y[i:(i + seq_length)]
            new_shape = (y_seq.shape[0], y_seq.shape[1] * (predict_labels))
            if IRI_or_RD == "IRI":
                y_seq_add = np.full((y_seq.shape[0], y_seq.shape[1] * (predict_labels - 1)), 100)
                new_seq = np.concatenate([x_seq, y_seq.reshape(-1, 1), y_seq_add], axis=1)
            else:
                y_seq_add_1 = np.full(1, 100)
                y_seq_add_other = np.full((y_seq.shape[0], y_seq.shape[1] * (predict_labels - 2)), 100)
                new_seq = np.concatenate([x_seq, y_seq_add_1.reshape(-1, 1), y_seq.reshape(-1, 1), y_seq_add_other.reshape(-1, 1)], axis=1)
            X.append(new_seq)
            Y.append(raw_y[i + seq_length])

    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32').reshape(X.shape[0], output_seq_len, n_out_features)
    return X, Y

def create_dataset_single(x, y, seq_length, output_seq_len=1, n_out_features=1):
    X, Y = [], []
    size = x.shape[0]
    for i in range(size - seq_length):
        X.append(x[i:i + seq_length, :])
        Y.append(y[i + seq_length])
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32').reshape(X.shape[0], output_seq_len, n_out_features)
    return X, Y

# =============================================================================
# Stage 1
# =============================================================================
import kerastuner as kt

class MyBayesianOptimization(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        kwargs["batch_size"] = hp.get("batch_size")
        super(MyBayesianOptimization, self).run_trial(trial, *args, **kwargs)

def build_hyper_model_M1(hp):
    bs = hp.Choice("batch_size", values=[32, 64, 128], default=64)
    learning_rate = hp.Float('shared_learning_rate', min_value=1e-6, max_value=1e-1, sampling='LOG', default=1e-2)
    encoder_inputs = Input(shape=(None, n_in_features + predict_labels), name='encoder_input')
    decoder_inputs = Input(shape=(None, n_out_features), name='decoder_input')
    Dense_activation = hp.Choice("shared_input_activation", values=["relu", "selu", "tanh"], default="relu")
    dense_units = hp.Int("shared_input_units", min_value=64, max_value=256, step=64, default=128)
    shared_feature_units = hp.Int("shared_input_units", min_value=10, max_value=210, step=50, default=10)
    shared_feature = Dense(dense_units, activation=Dense_activation, name='shared_feature_1')(encoder_inputs)
    shared_feature = tf.keras.layers.Dropout(0.3, name='shared_feature_dropout')(shared_feature)
    shared_feature = Dense(shared_feature_units, activation=Dense_activation, name='shared_feature')(shared_feature)
    y1_output, _ = build_task_branch(shared_feature, branch_name='y1_M1', n_out_features=n_out_features,
                                     hp=hp, prefix="task_", decoder_input=decoder_inputs)
    model = Model([encoder_inputs, decoder_inputs], y1_output)
    model.compile(Adam(learning_rate=learning_rate), loss='mean_squared_error')
    model._hp_batch_size = bs
    return model

tuner_M1 = kt.BayesianOptimization(
    build_hyper_model_M1,
    objective='val_loss',
    max_trials=max_trials_num,
    executions_per_trial=executions_per_trial_num,
    directory='hyper_tuning',
    project_name='M1_tuning',
    overwrite=False
)

raw_x_IRI, raw_y_IRI = load_data_single(f"txt_csv{os.sep}cleaned_result_array_IRI.txt")
scaler_x_IRI = MinMaxScaler().fit(raw_x_IRI)
scaler_y_IRI = StandardScaler().fit(raw_y_IRI)
x_IRI_scaled = scaler_x_IRI.transform(raw_x_IRI)
y_IRI_scaled = scaler_y_IRI.transform(raw_y_IRI)
X_IRI, Y_IRI = create_dataset_single_with_y_history(x_IRI_scaled, y_IRI_scaled, input_seq_len, n_out_features=n_out_features, IRI_or_RD = "IRI")
X_train_val1, X_test1, Y_train_val1, Y_test1 = train_test_split(
    X_IRI, Y_IRI, test_size=0.15, random_state=2022
)
X_train1, X_val1, Y_train1, Y_val1 = train_test_split(
    X_train_val1, Y_train_val1, test_size=15/85, random_state=2022
)
decoder_input_train1 = np.zeros_like(Y_train1)
decoder_input_val1 = np.zeros_like(Y_val1)
decoder_input_test1 = np.zeros_like(Y_test1)

print("\n---------- Tuning Stage 1: M1 (Y1 only) ----------")
checkpoint_filepath_M1 = os.path.join(output_folder, f"best_model_test{os.sep}best_model_M1.keras")
early_stop_M1 = EarlyStopping(monitor='val_loss', patience=patience_num, restore_best_weights=True)
model_checkpoint_M1 = ModelCheckpoint(filepath=checkpoint_filepath_M1, monitor='val_loss', save_best_only=True, save_weights_only=False)
print("M1 调优器搜索空间概要：")
tuner_M1.search_space_summary()
tuner_M1.search([X_train1, decoder_input_train1], Y_train1,
                epochs=epochs_num_hp, validation_data=([X_val1, decoder_input_val1], Y_val1),
                callbacks=[early_stop_M1, model_checkpoint_M1])
best_hp_M1 = tuner_M1.get_best_hyperparameters(num_trials=num_trials_num)[0]
print("M1 最优超参数：")
print(best_hp_M1.values)
plot_hyperparameter_search_best_epoch(tuner_M1, "Hyperparameter Tuning Loss (M1)", "Hyperparameter_Tuning_Loss_(M1).pdf")
best_model_M1 = tuner_M1.hypermodel.build(best_hp_M1)
best_model_M1.summary()
best_batch_size = best_hp_M1.get("batch_size")
print("最佳 batch_size：", best_batch_size)

history_M1 = best_model_M1.fit([X_train1, decoder_input_train1], Y_train1,
                               epochs=epochs_num_train, batch_size=best_batch_size,
                               validation_data=([X_test1, decoder_input_test1], Y_test1),
                               callbacks=[early_stop_M1, model_checkpoint_M1],
                               verbose=1)
plot_loss(history_M1, "Stage 1 Loss (M1: Y1 only)")
pred_IRI = best_model_M1.predict([X_val1, np.zeros(Y_val1.shape)])
true_pred = scaler_y_IRI.inverse_transform(pred_IRI.reshape(-1, 1))
true_y = scaler_y_IRI.inverse_transform(Y_val1.reshape(-1, 1))
evaluate(true_pred, true_y, "IRI")

# =============================================================================
# Stage 2
# =============================================================================
def auto_map_hp(old_hp, old_branch, new_branch):
    mapped = {}
    prefix_old = f"task_{old_branch}_"
    prefix_new = f"task_{new_branch}_"
    for key, value in old_hp.values.items():
        if key.startswith(prefix_old):
            new_key = prefix_new + key[len(prefix_old):]
            mapped[new_key] = value
    return mapped

class SimpleHP:
    def __init__(self, values):
        self.values = values
    def Choice(self, key, values, default):
        return self.values.get(key, default)
    def Int(self, key, min_value, max_value, step=1, default=None):
        return self.values.get(key, default)
    def Boolean(self, key, default):
        return self.values.get(key, default)
    def Float(self, key, min_value, max_value, sampling='LOG', default=None):
        return self.values.get(key, default)

def build_hyper_model_M2(hp_M2):
    base_model = best_model_M1

    for layer in best_model_M1.layers:
        if "shared" in layer.name:
            layer.trainable = False

    encoder_input, decoder_input = base_model.input
    shared_feature = base_model.get_layer('shared_feature').output
    shared_shape = tf.keras.backend.int_shape(shared_feature)[1:]
    shared_input = Input(shape=shared_shape, name="shared_input")
    decoder_input_y1 = Input(shape=(None, n_out_features), name='decoder_input_y1')
    mapped_hp_values = auto_map_hp(best_hp_M1, old_branch="y1_M1", new_branch="y1_M2")
    print("映射后的 Y1 分支超参数：", mapped_hp_values)
    mapped_hp = SimpleHP(mapped_hp_values)
    y1_branch, _ = build_task_branch(shared_input, branch_name='y1_M2', n_out_features=n_out_features,
                                    hp=mapped_hp, prefix="task_", decoder_input=decoder_input_y1)
    temp_y1 = Model([shared_input, decoder_input_y1], y1_branch, name="y1_M2_model")
    for layer in temp_y1.layers:
        if layer.name.startswith("y1_M2_"):
            suffix = layer.name[len("y1_M2_"):]
            try:
                old_layer = base_model.get_layer("y1_M1_" + suffix)
                layer.set_weights(old_layer.get_weights())
                layer.trainable = False
            except Exception as e:
                print(f"复制 Y1 分支 {suffix} 权重出错：{e}")
    y1_frozen = temp_y1([shared_feature, decoder_input])
    y2_output, _ = build_task_branch(shared_feature, branch_name='y2_M2', n_out_features=n_out_features,
                                    hp=hp_M2, prefix="task_", decoder_input=decoder_input)
    M2 = Model([encoder_input, decoder_input], [y1_frozen, y2_output], name="M2")
    learning_rate = hp_M2.Float('learning_rate_M2', min_value=1e-6, max_value=1e-1, sampling='LOG', default=1e-2)
    M2.compile(Adam(learning_rate=learning_rate), loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[0.3, 0.7])
    return M2

raw_x_RT, raw_y_RT = load_data_single(f"txt_csv{os.sep}cleaned_result_array_RD.txt")
scaler_x_RT = scaler_x_IRI
x_RT_scaled = scaler_x_RT.transform(raw_x_RT)
standard_scaler_y_RT = StandardScaler().fit(raw_y_RT)
raw_y_RT_trans = standard_scaler_y_RT.transform(raw_y_RT)
pt_RD = PowerTransformer(method='yeo-johnson')
y_RT_scaled = pt_RD.fit_transform(raw_y_RT_trans)

scaler_y_RT = StandardScaler().fit(raw_y_RT)
y_RT_scaled = scaler_y_RT.transform(raw_y_RT)
X_RT, Y_RT = create_dataset_single_with_y_history(x_RT_scaled, y_RT_scaled, input_seq_len)
X_train_val2, X_test2, Y_train_val2, Y_test2 = train_test_split(
    X_RT, Y_RT, test_size=0.15, random_state=2022
)

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(
    X_train_val2, Y_train_val2, test_size=15/85, random_state=2022
)
decoder_input_val2 = np.zeros_like(Y_val2)
decoder_input_train2 = np.zeros_like(Y_train2)
decoder_input_test2 = np.zeros_like(Y_test2)

tuner_M2 = kt.BayesianOptimization(
    lambda hp_M2: build_hyper_model_M2(hp_M2),
    objective='val_loss',
    max_trials=max_trials_num,
    executions_per_trial=executions_per_trial_num,
    directory='hyper_tuning',
    project_name='M2_tuning',
    overwrite=False
)
print("\n---------- Tuning Stage 2: M2 (Y2 only, Y1 branch frozen) ----------")
checkpoint_filepath_M2 = os.path.join(output_folder, f"best_model_test{os.sep}best_model_M2.keras")
early_stop_M2 = EarlyStopping(monitor='val_loss', patience=patience_num, min_delta=0, restore_best_weights=True)
model_checkpoint_M2 = ModelCheckpoint(filepath=checkpoint_filepath_M2, monitor='val_loss', save_best_only=True, save_weights_only=False)
dummy_Y_train2 = np.zeros_like(Y_train2)
dummy_Y_val2 = np.zeros_like(Y_val2)
print("M2 调优器搜索空间概要：")
tuner_M2.search_space_summary()
tuner_M2.search([X_train2, decoder_input_train2], [dummy_Y_train2, Y_train2],
                epochs=epochs_num_hp, batch_size=best_batch_size,
                validation_data=([X_val2, decoder_input_val2], [dummy_Y_val2, Y_val2]),
                callbacks=[early_stop_M2, model_checkpoint_M2])

best_hp_M2 = tuner_M2.get_best_hyperparameters(num_trials=num_trials_num)[0]
print("M2 最优超参数：")
print(best_hp_M2.values)
plot_hyperparameter_search_best_epoch(tuner_M2, "Hyperparameter Tuning Loss (M2)", "Hyperparameter_Tuning_Loss_(M2).pdf")
best_model_M2 = tuner_M2.hypermodel.build(best_hp_M2)
best_model_M2.summary()
history_M2 = best_model_M2.fit([X_train2, decoder_input_train2], [dummy_Y_train2, Y_train2],
                               epochs=epochs_num_train, batch_size=best_batch_size,
                               validation_data=([X_val2, decoder_input_val2], [dummy_Y_val2, Y_val2]),
                               callbacks=[early_stop_M2, model_checkpoint_M2],
                               verbose=0)
plot_loss(history_M2, "Stage 2 Loss (M2: Y2 only)")
pred_RT = best_model_M2.predict([X_val2, np.zeros(Y_val2.shape)])
def inverse_transform_RD(y_scaled):
    y_original = scaler_y_RT.inverse_transform(y_scaled)
    return y_original
true_pred_RT = inverse_transform_RD(pred_RT[1].reshape(-1, 1))
true_y_RT = inverse_transform_RD(Y_val2.reshape(-1, 1))
evaluate(true_pred_RT, true_y_RT, "RD")

def prepare_multi_task_labels(Y_list):
    Y_list_prepared = []
    sample_weights = []
    for y in Y_list:
        mask = (~np.isnan(y)).astype(np.float32)
        y_filled = np.nan_to_num(y, nan=0.0)
        Y_list_prepared.append(y_filled)
        sample_weights.append(mask.squeeze(axis=-1))
    return Y_list_prepared, sample_weights

# =============================================================================
# Stage 3
# =============================================================================
def build_hyper_model_M3(hp_M3):
    base_model = best_model_M2

    for layer in best_model_M1.layers:
        if "shared" in layer.name:
            layer.trainable = True

    encoder_input, decoder_input = base_model.input
    shared_feature = base_model.get_layer('shared_feature').output
    mapped_hp_values_M1 = auto_map_hp(best_hp_M1, old_branch="y1_M1", new_branch="y1_M3")
    mapped_hp_M1 = SimpleHP(mapped_hp_values_M1)
    shared_shape = tf.keras.backend.int_shape(shared_feature)[1:]
    shared_input = Input(shape=shared_shape, name="shared_input")
    decoder_input_y1 = Input(shape=(None, n_out_features), name='decoder_input_y1')
    y1_branch, _ = build_task_branch(shared_input, branch_name='y1_M3', n_out_features=n_out_features,
                                    hp=mapped_hp_M1, prefix="task_", decoder_input=decoder_input_y1)
    y1_model = Model([shared_input, decoder_input_y1], y1_branch, name="y1_M3_model")
    y1_M2_model_sub = base_model.get_layer("y1_M2_model")
    for layer in y1_model.layers:
        if layer.name.startswith("y1_M3_"):
            suffix = layer.name[len("y1_M3_"):]
            try:
                old_layer = y1_M2_model_sub.get_layer("y1_M2_" + suffix)
                layer.set_weights(old_layer.get_weights())
                layer.trainable = True
            except Exception as e:
                print(f"复制 Y1 分支 {suffix} 权重出错：{e}")
    y1_out = y1_model([shared_feature, decoder_input])
    mapped_hp_values_M2 = auto_map_hp(best_hp_M2, old_branch="y2_M2", new_branch="y2_M3")
    mapped_hp_M2 = SimpleHP(mapped_hp_values_M2)
    y2_branch, _ = build_task_branch(shared_feature, branch_name='y2_M3', n_out_features=n_out_features,
                                    hp=mapped_hp_M2, prefix="task_", decoder_input=decoder_input)
    y2_model = Model([encoder_input, decoder_input], y2_branch, name="y2_M3_model")
    for layer in y2_model.layers:
        if layer.name.startswith("y2_M3_"):
            suffix = layer.name[len("y2_M3_"):]
            try:
                old_layer = base_model.get_layer("y2_M2_" + suffix)
                layer.set_weights(old_layer.get_weights())
                layer.trainable = True
            except Exception as e:
                print(f"复制 Y2 分支 {suffix} 权重出错：{e}")
    y2_out = y2_model([encoder_input, decoder_input])
    additional_outputs = []
    num_additional = predict_labels - 2
    for idx in range(num_additional):
        branch, _ = build_task_branch(shared_feature, branch_name=f'y{idx + 3}', n_out_features=n_out_features,
                                      hp=hp_M3, prefix="task_", decoder_input=decoder_input)
        additional_outputs.append(branch)
    outputs = [y1_out, y2_out] + additional_outputs
    learning_rate = hp_M3.Float('learning_rate_M3', min_value=1e-6, max_value=1e-1, sampling='LOG', default=1e-2)
    M3 = Model([encoder_input, decoder_input], outputs, name="M3")
    M3.compile(Adam(learning_rate=learning_rate), loss=['mean_squared_error'] * (2 + num_additional))
    return M3

def load_data_all(filepath, n_in_features):
    dat = np.genfromtxt(filepath, delimiter=',')
    raw_x = dat[:, :n_in_features + 1]
    raw_y = dat[:, n_in_features + 1:]
    return raw_x, raw_y

def scale_targets(y):
    scaled_y = np.empty_like(y)
    scalers = []
    for j in range(y.shape[1]):
        col = y[:, j]
        valid_mask = ~np.isnan(col)
        if j == 0:
            scaler_y_all_IRI = StandardScaler()
            scaler_y_all_IRI.fit(col[valid_mask].reshape(-1, 1))
            col_scaled = scaler_y_all_IRI.transform(col.reshape(-1, 1)).flatten()
            col_scaled[~valid_mask] = np.nan
            scalers.append(scaler_y_all_IRI)
        elif j == 1:
            scaler_y_all_RT = StandardScaler()
            scaler_y_all_RT.fit(col[valid_mask].reshape(-1, 1))
            col_scaled = scaler_y_all_RT.transform(col.reshape(-1, 1)).flatten()
            col_scaled[~valid_mask] = np.nan
            scalers.append(scaler_y_all_RT)
        else:
            scaler = StandardScaler()
            scaler.fit(col[valid_mask].reshape(-1, 1))
            col_scaled = scaler.transform(col.reshape(-1, 1)).flatten()
            col_scaled[~valid_mask] = np.nan
            scalers.append(scaler)
        scaled_y[:, j] = col_scaled
    return scaled_y, scalers

def create_dataset_multi_with_y_history(x, y, seq_length, n_out_features=1):
    unique_ids = np.unique(x[:, 0])
    X, Y_list = [], [[] for _ in range(y.shape[1])]
    size = x.shape[0]
    for seg_id in unique_ids:
        mask = (x[:, 0] == seg_id)
        temp_dat = x[mask]
        num_cols = temp_dat.shape[1]
        raw_y = y[mask]
        raw_x = temp_dat[:, 1:num_cols]
        size = raw_y.shape[0]
        for i in range(size - seq_length):
            x_seq = raw_x[i:i + seq_length, :]
            y_seq = raw_y[i:i + seq_length, :]
            combined_seq = np.concatenate([x_seq, y_seq], axis=1)
            X.append(combined_seq)
            for j in range(y.shape[1]):
                Y_list[j].append(raw_y[i + seq_length, j])
    X = np.array(X, dtype='float32')
    Y_list = [np.array(yj, dtype='float32').reshape(X.shape[0], 1, n_out_features) for yj in Y_list]
    return X, Y_list

raw_x_all, raw_y_all = load_data_all(f"txt_csv{os.sep}cleaned_result_array_IRI_RDI_with_PCI.txt", n_in_features)
scaler_x_all = scaler_x_IRI
x_all_scaled = np.copy(raw_x_all)
scaler_x_all = MinMaxScaler().fit(raw_x_all[:, 1:])
x_all_scaled[:, 1:] = scaler_x_all.transform(raw_x_all[:, 1:])
raw_y_all_scaled, scalers_y_all = scale_targets(raw_y_all)
num_tasks = raw_y_all_scaled.shape[1]
X_all, Y_all_list = create_dataset_multi_with_y_history(x_all_scaled, raw_y_all_scaled, input_seq_len)

split1 = train_test_split(X_all, *Y_all_list, test_size=0.15, random_state=2024)
X_train_val3 = split1[0]
X_test3      = split1[1]

Y_train_val_list = []
Y_test_all       = []
for i in range(num_tasks):
    Y_train_val_list.append(split1[2 + i*2])
    Y_test_all.append(split1[2 + i*2 + 1])

split2 = train_test_split(X_train_val3, *Y_train_val_list, test_size=15/85, random_state=2024)
X_train3 = split2[0]
X_val3   = split2[1]

Y_train_all = []
Y_val_all   = []
for i in range(num_tasks):
    Y_train_all.append(split2[2 + i*2])
    Y_val_all.append(split2[2 + i*2 + 1])

Y_train_all_prepared, sample_weights_train = prepare_multi_task_labels(Y_train_all)
Y_val_all_prepared, sample_weights_val     = prepare_multi_task_labels(Y_val_all)

decoder_input_train3 = np.zeros_like(Y_train_all_prepared[0])
decoder_input_val3   = np.zeros_like(Y_val_all_prepared[0])
decoder_input_test3  = np.zeros_like(Y_test_all[0])

tuner_M3 = kt.BayesianOptimization(
    lambda hp_M3: build_hyper_model_M3(hp_M3),
    objective='val_loss',
    max_trials=max_trials_num,
    executions_per_trial=executions_per_trial_num,
    directory='hyper_tuning',
    project_name='M3_tuning',
    overwrite=False
)
print(f"\n---------- Tuning Stage 3: M3 (Multi-task: Y1~Y{predict_labels}) ----------")
checkpoint_filepath_M3 = os.path.join(output_folder, f"best_model_test{os.sep}best_model_M3.keras")
early_stop_M3 = EarlyStopping(monitor='val_loss', patience=patience_num, min_delta=0, restore_best_weights=True)
model_checkpoint_M3 = ModelCheckpoint(filepath=checkpoint_filepath_M3, monitor='val_loss', save_best_only=True, save_weights_only=False)
val_data = ([X_val3, decoder_input_val3], Y_val_all_prepared, sample_weights_val)
print("M3 调优器搜索空间概要：")
tuner_M3.search_space_summary()
tuner_M3.search([X_train3, decoder_input_train3], Y_train_all_prepared,
                epochs=epochs_num_hp, batch_size=best_batch_size,
                validation_data=val_data,
                callbacks=[early_stop_M3, model_checkpoint_M3],
                verbose=1)
best_hp_M3 = tuner_M3.get_best_hyperparameters(num_trials=num_trials_num)[0]
print("M3 最优超参数：")
print(best_hp_M3.values)
plot_hyperparameter_search_best_epoch(tuner_M3, "Hyperparameter Tuning Loss (M3)", "Hyperparameter_Tuning_Loss_(M3).pdf")
best_model_M3 = tuner_M3.hypermodel.build(best_hp_M3)
best_model_M3.summary()

history_M3 = best_model_M3.fit([X_train3, decoder_input_train3], Y_train_all_prepared,
                               sample_weight=sample_weights_train,
                               epochs=epochs_num_train, batch_size=best_batch_size,
                               validation_data=([X_val3, decoder_input_val3], Y_val_all_prepared, sample_weights_val),
                               callbacks=[early_stop_M3, model_checkpoint_M3],
                               verbose=1)
plot_loss(history_M3, "Stage 3 Loss (M3: Multi-task)")
pred_M3 = best_model_M3.predict([X_val3, np.zeros(Y_val_all_prepared[0].shape)])
index_list = ["IRI_M3", "RD_M3", "PCI_M3"]
for i in range(len(pred_M3)):
    print(f"\n--- Evaluation for Task Y{i + 1} (index: {index_list[i]}) ---")
    true_pred_M3 = scalers_y_all[i].inverse_transform(pred_M3[i].reshape(-1, 1))
    true_y_M3 = scalers_y_all[i].inverse_transform(Y_val_all_prepared[i].reshape(-1, 1))
    evaluate(true_pred_M3, true_y_M3, f"{index_list[i]}", sample_weight=sample_weights_val[i])

if __name__ == '__main__':
    print("程序运行开始...")
    print("程序运行结束。")
    log_file.close()
