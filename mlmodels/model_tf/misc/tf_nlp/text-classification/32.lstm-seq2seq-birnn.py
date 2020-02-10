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
        batch_size,
        size_layer,
        num_layers,
        embedded_size,
        dict_size,
        dimension_output,
        learning_rate,
    ):
        def cells(size, reuse=False):
            return tf.nn.rnn_cell.LSTMCell(
                size, initializer=tf.orthogonal_initializer(), reuse=reuse
            )

        self.X = tf.placeholder(tf.int32, [None, None])
        self.X_DEC = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.X_DEC_seq_len = tf.placeholder(tf.int32, [None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])

        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        decoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X_DEC)

        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells(size_layer // 2),
                cell_bw=cells(size_layer // 2),
                inputs=encoder_embedded,
                sequence_length=self.X_seq_len,
                dtype=tf.float32,
                scope="bidirectional_rnn_%d" % (n),
            )
            encoder_embedded = tf.concat((out_fw, out_bw), 2)
        bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
        bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)
        bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
        encoder_state = tuple([bi_lstm_state] * num_layers)

        decoder_cells = tf.nn.rnn_cell.MultiRNNCell([cells(size_layer) for _ in range(num_layers)])
        dense_layer = tf.layers.Dense(dimension_output)
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoder_embedded, sequence_length=self.X_DEC_seq_len, time_major=False
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cells,
            helper=training_helper,
            initial_state=encoder_state,
            output_layer=dense_layer,
        )
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            impute_finished=True,
            maximum_iterations=tf.reduce_max(self.X_DEC_seq_len),
        )
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=encoder_embeddings,
            start_tokens=tf.tile(tf.constant([GO], dtype=tf.int32), [batch_size]),
            end_token=EOS,
        )
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cells,
            helper=predicting_helper,
            initial_state=encoder_state,
            output_layer=dense_layer,
        )
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=predicting_decoder,
            impute_finished=False,
            maximum_iterations=tf.reduce_max(self.X_seq_len),
        )
        self.logits = training_decoder_output.rnn_output[:, -1]
        self.predicting_ids = predicting_decoder_output.sample_id[:, -1]
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.cast(self.predicting_ids, tf.int64), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[7]:


size_layer = 128
num_layers = 2
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128
skip = 5


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    batch_size,
    size_layer,
    num_layers,
    embedded_size,
    vocabulary_size + 4,
    dimension_output,
    learning_rate,
)
sess.run(tf.global_variables_initializer())


# In[9]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:%d\n" % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (len(train_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(train_X[i : i + batch_size], dictionary, maxlen)
        acc, loss, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={
                model.X: batch_x,
                model.X_DEC: batch_x[:, skip:],
                model.X_seq_len: [maxlen] * batch_size,
                model.X_DEC_seq_len: [maxlen - skip] * batch_size,
                model.Y: train_onehot[i : i + batch_size],
            },
        )
        train_loss += loss
        train_acc += acc

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(test_X[i : i + batch_size], dictionary, maxlen)
        acc, loss = sess.run(
            [model.accuracy, model.cost],
            feed_dict={
                model.X: batch_x,
                model.X_DEC: batch_x[:, skip:],
                model.X_seq_len: [maxlen] * batch_size,
                model.X_DEC_seq_len: [maxlen - skip] * batch_size,
                model.Y: test_onehot[i : i + batch_size],
            },
        )
        test_loss += loss
        test_acc += acc

    train_loss /= len(train_X) // batch_size
    train_acc /= len(train_X) // batch_size
    test_loss /= len(test_X) // batch_size
    test_acc /= len(test_X) // batch_size

    if test_acc > CURRENT_ACC:
        print("epoch: %d, pass acc: %f, current acc: %f" % (EPOCH, CURRENT_ACC, test_acc))
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
    else:
        CURRENT_CHECKPOINT += 1

    print("time taken:", time.time() - lasttime)
    print(
        "epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n"
        % (EPOCH, train_loss, train_acc, test_loss, test_acc)
    )
    EPOCH += 1


# In[ ]:
