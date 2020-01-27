#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

# In[2]:


def build_dataset(words, n_words):
    count = [["GO", 0], ["PAD", 1], ["EOS", 2], ["UNK", 3]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def str_idx(corpus, dic, maxlen, UNK=3):
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i][:maxlen][::-1]):
            val = dic[k] if k in dic else UNK
            X[i, -1 - no] = val
    return X


def load_data(filepath):
    x1 = []
    x2 = []
    y = []
    for line in open(filepath):
        l = line.strip().split("\t")
        if len(l) < 2:
            continue
        if random.random() > 0.5:
            x1.append(l[0].lower())
            x2.append(l[1].lower())
        else:
            x1.append(l[1].lower())
            x2.append(l[0].lower())
        y.append(1)
    combined = np.asarray(x1 + x2)
    shuffle_indices = np.random.permutation(np.arange(len(combined)))
    combined_shuff = combined[shuffle_indices]
    for i in range(len(combined)):
        x1.append(combined[i])
        x2.append(combined_shuff[i])
        y.append(0)
    return np.array(x1), np.array(x2), np.array(y)


# In[3]:


X1_text, X2_text, Y = load_data("person_match.train")


# In[4]:


concat = " ".join(X1_text.tolist() + X2_text.tolist())
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[5]:


def _pairwise_distances(embeddings_left, embeddings_right, squared=False):
    dot_product = tf.matmul(embeddings_left, tf.transpose(embeddings_right))
    square_norm = tf.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings_left, embeddings_right, margin, squared=False):
    pairwise_dist = _pairwise_distances(embeddings_left, embeddings_right, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


# In[6]:


class Model:
    def __init__(
        self, size_layer, num_layers, embedded_size, dict_size, learning_rate, dimension_output
    ):
        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(
                size_layer, initializer=tf.orthogonal_initializer(), reuse=reuse
            )

        def rnn(inputs, reuse=False):
            with tf.variable_scope("model", reuse=reuse):
                rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
                outputs, _ = tf.nn.dynamic_rnn(rnn_cells, inputs, dtype=tf.float32)
                return tf.layers.dense(outputs[:, -1], dimension_output)

        self.X_left = tf.placeholder(tf.int32, [None, None])
        self.X_right = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None])
        self.batch_size = tf.shape(self.X_left)[0]
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        embedded_left = tf.nn.embedding_lookup(encoder_embeddings, self.X_left)
        embedded_right = tf.nn.embedding_lookup(encoder_embeddings, self.X_right)

        self.output_left = rnn(embedded_left, False)
        self.output_right = rnn(embedded_right, True)

        self.cost, fraction = batch_all_triplet_loss(
            self.Y, self.output_left, self.output_right, margin=0.5, squared=False
        )

        self.distance = tf.sqrt(
            tf.reduce_sum(
                tf.square(tf.subtract(self.output_left, self.output_right)), 1, keep_dims=True
            )
        )
        self.distance = tf.div(
            self.distance,
            tf.add(
                tf.sqrt(tf.reduce_sum(tf.square(self.output_left), 1, keep_dims=True)),
                tf.sqrt(tf.reduce_sum(tf.square(self.output_right), 1, keep_dims=True)),
            ),
        )
        self.distance = tf.reshape(self.distance, [-1])

        self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance))
        correct_predictions = tf.equal(self.temp_sim, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


# In[7]:


size_layer = 256
num_layers = 2
embedded_size = 128
learning_rate = 1e-3
dimension_output = 300
maxlen = 30
batch_size = 128


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    size_layer, num_layers, embedded_size, len(dictionary), learning_rate, dimension_output
)
sess.run(tf.global_variables_initializer())


# In[9]:


vectors_left = str_idx(X1_text, dictionary, maxlen)
vectors_right = str_idx(X2_text, dictionary, maxlen)
train_X_left, test_X_left, train_X_right, test_X_right, train_Y, test_Y = train_test_split(
    vectors_left, vectors_right, Y, test_size=0.2
)


# In[10]:


for EPOCH in range(5):
    lasttime = time.time()

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    pbar = tqdm(range(0, len(train_X_left), batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x_left = train_X_left[i : min(i + batch_size, train_X_left.shape[0])]
        batch_x_right = train_X_right[i : min(i + batch_size, train_X_left.shape[0])]
        batch_y = train_Y[i : min(i + batch_size, train_X_left.shape[0])]
        acc, loss, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X_left: batch_x_left, model.X_right: batch_x_right, model.Y: batch_y},
        )
        assert not np.isnan(loss)
        train_loss += loss
        train_acc += acc
        pbar.set_postfix(cost=loss, accuracy=acc)

    pbar = tqdm(range(0, len(test_X_left), batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x_left = test_X_left[i : min(i + batch_size, train_X_left.shape[0])]
        batch_x_right = test_X_right[i : min(i + batch_size, train_X_left.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, train_X_left.shape[0])]
        acc, loss = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.X_left: batch_x_left, model.X_right: batch_x_right, model.Y: batch_y},
        )
        test_loss += loss
        test_acc += acc
        pbar.set_postfix(cost=loss, accuracy=acc)

    train_loss /= len(train_X_left) / batch_size
    train_acc /= len(train_X_left) / batch_size
    test_loss /= len(test_X_left) / batch_size
    test_acc /= len(test_X_left) / batch_size

    print("time taken:", time.time() - lasttime)
    print(
        "epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n"
        % (EPOCH, train_loss, train_acc, test_loss, test_acc)
    )


# In[11]:


left = str_idx(["adriana evans"], dictionary, maxlen)
right = str_idx(["adriana"], dictionary, maxlen)
sess.run([model.temp_sim, 1 - model.distance], feed_dict={model.X_left: left, model.X_right: right})


# In[12]:


left = str_idx(["husein zolkepli"], dictionary, maxlen)
right = str_idx(["zolkepli"], dictionary, maxlen)
sess.run([model.temp_sim, 1 - model.distance], feed_dict={model.X_left: left, model.X_right: right})


# In[13]:


left = str_idx(["adriana evans"], dictionary, maxlen)
right = str_idx(["evans adriana"], dictionary, maxlen)
sess.run([model.temp_sim, 1 - model.distance], feed_dict={model.X_left: left, model.X_right: right})


# In[15]:


left = str_idx(["synergy telecom"], dictionary, maxlen)
right = str_idx(["syntel"], dictionary, maxlen)
sess.run([model.temp_sim, 1 - model.distance], feed_dict={model.X_left: left, model.X_right: right})


# In[ ]:
