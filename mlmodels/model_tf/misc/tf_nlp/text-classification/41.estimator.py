#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import tensorflow as tf
from sklearn.cross_validation import train_test_split

from utils import *

# In[2]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
trainset.data = trainset.data[:1000]
trainset.target = trainset.target[:1000]
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[3]:


ONEHOT = np.zeros((len(trainset.data), len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)), trainset.target] = 1.0
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(
    trainset.data, trainset.target, ONEHOT, test_size=0.2
)


# In[4]:


concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[5]:


GO = dictionary["GO"]
PAD = dictionary["PAD"]
EOS = dictionary["EOS"]
UNK = dictionary["UNK"]


# In[6]:


class Model:
    def __init__(
        self,
        size_layer,
        num_layers,
        embedded_size,
        batch_size,
        from_dict_size,
        to_dict_size,
        grad_clip=5.0,
    ):
        self.size_layer = size_layer
        self.num_layers = num_layers
        self.embedded_size = embedded_size
        self.grad_clip = grad_clip
        self.from_dict_size = from_dict_size
        self.to_dict_size = to_dict_size
        self.batch_size = batch_size
        self.model = tf.estimator.Estimator(self.model_fn)

    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.size_layer, reuse=reuse)

    def seq2seq(self, x_dict):
        x = x_dict["x"]
        x_seq_len = x_dict["x_len"]
        with tf.variable_scope("encoder"):
            encoder_embedding = tf.get_variable(
                "encoder_embedding",
                [self.from_dict_size, self.embedded_size],
                tf.float32,
                tf.random_uniform_initializer(-1.0, 1.0),
            )
            _, encoder_state = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.MultiRNNCell(
                    [self.lstm_cell() for _ in range(self.num_layers)]
                ),
                inputs=tf.nn.embedding_lookup(encoder_embedding, x),
                sequence_length=x_seq_len,
                dtype=tf.float32,
            )
            encoder_state = tuple(encoder_state[-1] for _ in range(self.num_layers))
        y = x_dict["y"]
        y_seq_len = x_dict["y_len"]
        with tf.variable_scope("decoder"):
            decoder_embedding = tf.get_variable(
                "decoder_embedding",
                [self.to_dict_size, self.embedded_size],
                tf.float32,
                tf.random_uniform_initializer(-1.0, 1.0),
            )
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.nn.embedding_lookup(decoder_embedding, y),
                sequence_length=y_seq_len,
                time_major=False,
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=tf.nn.rnn_cell.MultiRNNCell(
                    [self.lstm_cell() for _ in range(self.num_layers)]
                ),
                helper=helper,
                initial_state=encoder_state,
                output_layer=tf.layers.Dense(len(trainset.target_names)),
            )
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, impute_finished=True, maximum_iterations=tf.reduce_max(y_seq_len)
            )
            return decoder_output.rnn_output[:, -1]

    def model_fn(self, features, labels, mode):
        logits = self.seq2seq(features)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=logits)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        params = tf.trainable_variables()
        gradients = tf.gradients(cost, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        train_op = tf.train.AdamOptimizer().apply_gradients(
            zip(clipped_gradients, params), global_step=tf.train.get_global_step()
        )
        acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(logits, 1))
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=tf.argmax(logits, 1),
            loss=cost,
            train_op=train_op,
            eval_metric_ops={"accuracy": acc_op},
        )
        return estim_specs


# In[7]:


size_layer = 256
num_layers = 2
embedded_size = 256
batch_size = len(train_X)
maxlen = 50
skip = 5
model = Model(
    size_layer, num_layers, embedded_size, batch_size, vocabulary_size + 4, vocabulary_size + 4
)


# In[8]:


batch_x = str_idx(train_X, dictionary, maxlen).astype(np.int32)
batch_y = batch_x[:, skip:]
seq_x = np.array([maxlen] * len(train_X)).astype(np.int32)
seq_y = np.array([maxlen - skip] * len(train_X)).astype(np.int32)


# In[9]:


input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": batch_x, "x_len": seq_x, "y": batch_y, "y_len": seq_y},
    y=train_onehot,
    batch_size=batch_size,
    num_epochs=10,
    shuffle=False,
)
model.model.train(input_fn)


# In[ ]:
