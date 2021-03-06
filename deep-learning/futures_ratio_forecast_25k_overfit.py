import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')
	
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
sns.set()

df = pd.read_excel('../dataset/hchi_ratio_25k.xlsx')
df = df.reset_index()
df.head()

TEST_SIZE = 1440 # minutes
DATASET_SIZE = df.shape[0] # rows

minmax = MinMaxScaler().fit(df.loc[:DATASET_SIZE - TEST_SIZE - 1, 'HCHI'].astype('float32').values.reshape(-1, 1)) # select HCHI ratio column
df_log = minmax.transform(df.loc[:DATASET_SIZE - TEST_SIZE - 1, 'HCHI'].astype('float32').values.reshape(-1, 1)) # select HCHI ratio column
df_log = pd.DataFrame(df_log)
df_log.head()
df_log.shape

simulation_size = 1
num_layers = 1
size_layer = 128
timestamp = 5
epoch = 300
dropout_rate = 1
test_size = TEST_SIZE
learning_rate = 0.01

df_train = df_log
df.shape, df_train.shape

class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
    ):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        drop = tf.contrib.rnn.DropoutWrapper(
            rnn_cells, output_keep_prob = forget_bias
        )
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
        
def calculate_accuracy(real, predict):
    real = np.array(real)
    predict = np.array(predict)
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
	
from tensorflow.python.framework import ops

def forecast():
    ops.reset_default_graph()
#     tf.reset_default_graph()
    modelnn = Model(
        learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate
    )
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

    pbar = tqdm(range(epoch), desc = 'train loop')
    last_cost = 0.0
    for i in pbar:
        init_value = np.zeros((1, num_layers * 2 * size_layer))
        total_loss, total_acc = [], []
        for k in range(0, df_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x = np.expand_dims(
                df_train.iloc[k : index, :].values, axis = 0
            )
            batch_y = df_train.iloc[k + 1 : index + 1, :].values
            logits, last_state, _, loss = sess.run(
                [modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],
                feed_dict = {
                    modelnn.X: batch_x,
                    modelnn.Y: batch_y,
                    modelnn.hidden_layer: init_value,
                },
            )        
            init_value = last_state
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
        current_cost = np.mean(total_loss)
        pbar.set_postfix(cost = current_cost, acc = np.mean(total_acc))
        if current_cost == last_cost:
            break
        last_cost = current_cost
    
    future_day = test_size

    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    upper_b = (df_train.shape[0] // timestamp) * timestamp
    init_value = np.zeros((1, num_layers * 2 * size_layer))

    for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(
                    df_train.iloc[k : k + timestamp], axis = 0
                ),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[k + 1 : k + timestamp + 1] = out_logits

    if upper_b != df_train.shape[0]:
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis = 0),
                modelnn.hidden_layer: init_value,
            },
        )
        output_predict[upper_b + 1 : df_train.shape[0] + 1] = out_logits
        future_day -= 1
        date_ori.append(date_ori[-1] + timedelta(days = 1))

    init_value = last_state
    
    for i in range(future_day):
        o = output_predict[-future_day - timestamp + i:-future_day + i]
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(o, axis = 0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[-future_day + i] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    
    output_predict = minmax.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.4)
    
    return deep_future
	
results = []
for i in range(simulation_size):
    print('simulation %d'%(i + 1))
    results.append(forecast())
	
date_ori = df.loc[:, 'dates']
date_ori[-5:]

accepted_results = []
for r in results:
    if (np.array(r[-test_size:]) < np.min(df['HCHI'])).sum() == 0 and \
    (np.array(r[-test_size:]) > np.max(df['HCHI']) * 2).sum() == 0:
        accepted_results.append(r)
len(accepted_results)

results_frame = pd.DataFrame(accepted_results)
results_frame = results_frame.transpose() 

results_frame['Dates'] = date_ori
results_frame['Avg'] = np.mean(results_frame, axis=1)
cols = results_frame.columns.tolist()
cols = cols[-2:] + cols[:-2] # reorder columns
results_frame = results_frame[cols] 

results_frame.to_excel('./HCHI_predictions_25k_overfit.xlsx')
results_frame.head

train_accuracies = [calculate_accuracy(df.loc[:DATASET_SIZE - TEST_SIZE - 1, 'HCHI'].values, r[:DATASET_SIZE - TEST_SIZE]) for r in accepted_results]
test_accuracies = [calculate_accuracy(df.loc[DATASET_SIZE - TEST_SIZE:, 'HCHI'].values, r[DATASET_SIZE - TEST_SIZE:]) for r in accepted_results]
avg_train_acc = np.mean(train_accuracies)
avg_test_acc = np.mean(test_accuracies)
accuracy_statement = 'average train accuracy: {0:.4f} \n average test accuracy: {1:.4f}'.format(avg_train_acc, avg_test_acc)
print(accuracy_statement)

plt.figure(figsize = (15, 5))
for no, r in enumerate(accepted_results):
    plt.plot(r, label = 'forecast %d'%(no + 1))
plt.plot(df['HCHI'], label = 'true trend', c = 'black')
plt.legend()
plt.title(accuracy_statement)

x_range_future = np.arange(len(results[0]))
tick_size = int(DATASET_SIZE / 20)
plt.xticks(x_range_future[::tick_size], date_ori[::tick_size])
plt.gcf().autofmt_xdate()
plt.savefig('HCHI_predictions_25k_overfit.png')
