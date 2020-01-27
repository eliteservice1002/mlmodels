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
        num_layers,
        embedded_size,
        dict_size,
        dimension_output,
        maxlen,
        grad_clip=5.0,
        kernel_sizes=[3, 3, 3],
    ):
        n_filters = [25 * k for k in kernel_sizes]

        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(
                size_layer, initializer=tf.orthogonal_initializer(), reuse=reuse
            )

        def add_highway(x, i):
            size = sum(n_filters)
            reshaped = tf.reshape(x, [-1, size])
            H = tf.layers.dense(reshaped, size, tf.nn.relu, name="activation" + str(i))
            T = tf.layers.dense(reshaped, size, tf.sigmoid, name="transform_gate" + str(i))
            C = tf.subtract(1.0, T)
            highway_out = tf.add(tf.multiply(H, T), tf.multiply(reshaped, C))
            return tf.reshape(highway_out, [-1, 1, size])

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        encoder_embedded = tf.reshape(encoder_embedded, [-1, maxlen, embedded_size])

        parallels = []
        for i, (n_filter, kernel_size) in enumerate(zip(n_filters, kernel_sizes)):
            conv_out = tf.layers.conv1d(
                inputs=encoder_embedded,
                filters=n_filter,
                kernel_size=kernel_size,
                activation=tf.tanh,
                name="conv1d" + str(i),
            )
            pool_out = tf.layers.max_pooling1d(
                inputs=conv_out, pool_size=conv_out.get_shape().as_list()[1], strides=1
            )
            parallels.append(tf.reshape(pool_out, [-1, n_filter]))
        pointer = tf.concat(parallels, 1)

        for i in range(2):
            pointer = add_highway(pointer, i)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        outputs, _ = tf.nn.dynamic_rnn(rnn_cells, pointer, dtype=tf.float32)
        W = tf.get_variable(
            "w", shape=(size_layer, dimension_output), initializer=tf.orthogonal_initializer()
        )
        b = tf.get_variable("b", shape=(dimension_output), initializer=tf.zeros_initializer())
        self.logits = tf.matmul(outputs[:, -1], W) + b
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
num_layers = 2
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(size_layer, num_layers, embedded_size, vocabulary_size + 4, dimension_output, maxlen)
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


# In[10]:


logits = sess.run(model.logits, feed_dict={model.X: str_idx(test_X, dictionary, maxlen)})
print(
    metrics.classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names)
)


# In[ ]:
