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
        size_layer,
        embedded_size,
        maxlen,
        batch_size,
        dict_size,
        dimension_output,
        grad_clip=5.0,
    ):
        def cells(reuse=False):
            return tf.nn.rnn_cell.GRUCell(
                size_layer, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse
            )

        self.X = tf.placeholder(tf.int32, [None, maxlen])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        enc_rnn_out, enc_state = tf.nn.dynamic_rnn(
            cell=cells(), inputs=encoder_embedded, dtype=tf.float32
        )

        def loop_fn(state, masks):
            query = tf.expand_dims(state, -1)
            align = tf.squeeze(tf.matmul(enc_rnn_out, query), -1)
            return align * masks

        def point(idx):
            idx = tf.expand_dims(idx, 1)
            b = tf.expand_dims(tf.range(batch_size), 1)
            c = tf.concat((tf.to_int64(b), idx), 1)
            return tf.gather_nd(encoder_embedded, c)

        starts = tf.fill([batch_size], GO)
        inp = tf.nn.embedding_lookup(encoder_embeddings, starts)
        masks = tf.to_float(tf.sign(self.X))
        outputs = []
        cell = cells()
        for i in range(maxlen):
            _, state = cell(inp, enc_state)
            output = loop_fn(state, masks)
            outputs.append(output)
            idx = tf.argmax(output, -1)
            inp = point(idx)
        outputs = tf.stack(outputs, 1)
        self.logits = tf.layers.dense(outputs, dimension_output)[:, -1]
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        params = tf.trainable_variables()
        gradients = tf.gradients(self.cost, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
        self.optimizer = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[7]:


size_layer = 128
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(size_layer, embedded_size, maxlen, batch_size, vocabulary_size + 4, dimension_output)
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
            feed_dict={model.X: batch_x, model.Y: train_onehot[i : i + batch_size]},
        )
        train_loss += loss
        train_acc += acc

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(test_X[i : i + batch_size], dictionary, maxlen)
        acc, loss = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.X: batch_x, model.Y: test_onehot[i : i + batch_size]},
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
