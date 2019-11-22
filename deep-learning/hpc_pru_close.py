import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

# sns.set()
# tf.compat.v1.random.set_random_seed(1234)

COMPANY_NAME = 'pru'
COLUMN_USED = 'Close'
DAYS_TO_PREDICT = 3

df = pd.read_csv('../dataset/{}.csv'.format(COMPANY_NAME))
df.head()

if COLUMN_USED is 'Close':
    col_index = [4, 5] # Close index
elif COLUMN_USED is 'High':
    col_index = [2, 3] # High index
else:
    col_index = [3, 4] # Low index

minmax = MinMaxScaler().fit(df.iloc[:-DAYS_TO_PREDICT, col_index[0]: col_index[1]].astype('float32'))
df_log = minmax.transform(df.iloc[:-DAYS_TO_PREDICT, col_index[0]: col_index[1]].astype('float32'))
true_values = np.array(df.iloc[-DAYS_TO_PREDICT:, col_index[0]: col_index[1]])

df_log = pd.DataFrame(df_log)
df_log.head()

simulation_size = 10
num_layers = 1
size_layer = 128
timestamp = 5
epoch = 300
dropout_rate = 0.8
test_size = DAYS_TO_PREDICT
learning_rate = 0.01

df_train = df_log
print(df.shape, df_train.shape)


class Model:
    def __init__(
            self,
            learning_rate,
            num_layers,
            size,
            size_layer,
            output_size,
            forget_bias=0.1,
    ):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple=False,
        )
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        drop = tf.contrib.rnn.DropoutWrapper(
            rnn_cells, output_keep_prob=forget_bias
        )
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state=self.hidden_layer, dtype=tf.float32
        )
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100


def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer


def forecast():
    tf.reset_default_graph()
    modelnn = Model(
        learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate
    )
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

    pbar = tqdm(range(epoch), desc='train loop')
    for i in pbar:
        init_value = np.zeros((1, num_layers * 2 * size_layer))
        total_loss, total_acc = [], []
        for k in range(0, df_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x = np.expand_dims(
                df_train.iloc[k: index, :].values, axis=0
            )
            batch_y = df_train.iloc[k + 1: index + 1, :].values
            logits, last_state, _, loss = sess.run(
                [modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],
                feed_dict={
                    modelnn.X: batch_x,
                    modelnn.Y: batch_y,
                    modelnn.hidden_layer: init_value,
                },
            )
            init_value = last_state
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
        pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))

    future_day = test_size

    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    upper_b = (df_train.shape[0] // timestamp) * timestamp
    init_value = np.zeros((1, num_layers * 2 * size_layer))

    for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict={
                modelnn.X: np.expand_dims(
                    df_train.iloc[k: k + timestamp], axis=0
                ),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[k + 1: k + timestamp + 1] = out_logits

    if upper_b != df_train.shape[0]:
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict={
                modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis=0),
                modelnn.hidden_layer: init_value,
            },
        )
        output_predict[upper_b + 1: df_train.shape[0] + 1] = out_logits
        future_day -= 1
        date_ori.append(date_ori[-1] + timedelta(days=1))

    init_value = last_state

    for i in range(future_day):
        o = output_predict[-future_day - timestamp + i:-future_day + i]
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict={
                modelnn.X: np.expand_dims(o, axis=0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[-future_day + i] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(days=1))

    output_predict = minmax.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.4)

    return deep_future

results = []
for i in range(simulation_size):
    print('simulation %d'%(i + 1))
    results.append(forecast())

date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
for i in range(test_size):
    date_ori.append(date_ori[-1] + timedelta(days = 1))
date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
print(date_ori[-5:])

accepted_results = []
for r in results:
    if (np.array(r[-test_size:]) < np.min(df[COLUMN_USED])).sum() == 0 and \
    (np.array(r[-test_size:]) > np.max(df[COLUMN_USED]) * 2).sum() == 0:
        accepted_results.append(r)
len(accepted_results)

accuracies = [calculate_accuracy(df[COLUMN_USED].values, r[:-test_size]) for r in accepted_results]
print('average accuracy: %.4f'%(np.mean(accuracies)))

# plt.figure(figsize = (15, 5))
# for no, r in enumerate(accepted_results):
#     plt.plot(r, label = 'forecast %d'%(no + 1))
# # plt.plot(df[COLUMN_USED], label = 'true trend', c = 'black')
# plt.legend()
# plt.title('average accuracy: %.4f'%(np.mean(accuracies)))

x_range_future = np.arange(len(results[0]))
# plt.xticks(x_range_future[::30], date_ori[::30])

# plt.show()

accepted_results_arr = np.array(accepted_results)

last_3_days = accepted_results_arr[:, -DAYS_TO_PREDICT:]
print('last_3_days:')
print(last_3_days)

one_day_errors = []
for index, value in enumerate(true_values):
    average = np.average(last_3_days[:, index])
    one_day_error = np.absolute((value - average) / value)
    one_day_errors.append(one_day_error)

print('one_day_errors:')
print(one_day_errors)

one_day_errors_raw = []
for index, value in enumerate(true_values):
    average = np.average(last_3_days[:, index])
    one_day_error_raw = (value - average) / value
    one_day_errors_raw.append(one_day_error_raw)

print('one_day_errors_raw:')
print(one_day_errors_raw)

three_day_avg_error = np.average(one_day_errors)
three_day_avg_error_raw = np.average(one_day_errors_raw)

print('three_day_avg_error:')
print(three_day_avg_error)
print('three_day_avg_error_raw:')
print(three_day_avg_error_raw)

three_day_exp_error = (one_day_errors[0] + one_day_errors[1] * 2 + one_day_errors[2] * 3) / 6
three_day_exp_error_raw = (one_day_errors_raw[0] + one_day_errors_raw[1] * 2 + one_day_errors_raw[2] * 3) / 6

print('three_day_exp_error:')
print(three_day_exp_error)
print('three_day_exp_error_raw:')
print(three_day_exp_error_raw)

results_frame = pd.DataFrame(accepted_results)
results_frame.to_csv('./{}_{}_predictions.csv'.format(COMPANY_NAME, COLUMN_USED))
