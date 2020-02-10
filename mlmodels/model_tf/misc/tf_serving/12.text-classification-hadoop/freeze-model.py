#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import json
import os
import re
import time

import numpy as np
import sklearn.datasets
import tensorflow as tf
from sklearn import metrics
from sklearn.cross_validation import train_test_split

# In[2]:


def clearstring(string):
    string = re.sub("[^A-Za-z ]+", " ", string.lower())
    string = string.split(" ")
    string = filter(None, string)
    return " ".join(string)


def separate_dataset(trainset, ratio=1):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split("\n")
        data_ = list(filter(None, data_))
        for n in range(int(len(data_) * ratio)):
            datastring.append(clearstring(data_[n]))
        for n in range(int(len(data_) * ratio)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


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
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            try:
                X[i, -1 - no] = dic[k]
            except Exception as e:
                X[i, -1 - no] = UNK
    return X


# In[3]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, ratio=0.05)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[4]:


ONEHOT = np.zeros((len(trainset.data), len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)), trainset.target] = 1.0
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(
    trainset.data, trainset.target, ONEHOT, test_size=0.2
)


# In[5]:


concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[6]:


GO = dictionary["GO"]
PAD = dictionary["PAD"]
EOS = dictionary["EOS"]
UNK = dictionary["UNK"]


# In[7]:


class Model:
    def __init__(
        self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate
    ):
        def cells(reuse=False):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer, reuse=reuse)

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        outputs, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)
        rnn_W = tf.Variable(tf.random_normal((size_layer, dimension_output)))
        rnn_B = tf.Variable(tf.random_normal([dimension_output]))
        # put 'logits' name is very important
        self.logits = tf.add(tf.matmul(outputs[:, -1], rnn_W), rnn_B, name="logits")
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[8]:


size_layer = 64
num_layers = 1
embedded_size = 64
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 32


# In[9]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    size_layer, num_layers, embedded_size, vocabulary_size + 4, dimension_output, learning_rate
)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
saver.save(sess, os.getcwd() + "/model-test/test")


# In[10]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:%d\n" % (EPOCH))
        saver.save(sess, os.getcwd() + "/model-test/test")
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


# In[11]:


strings = ",".join(
    [
        n.name
        for n in tf.get_default_graph().as_graph_def().node
        if "Variable" in n.op or n.name.find("Placeholder") >= 0 or n.name.find("logits") == 0
    ]
)


# In[12]:


def freeze_graph(model_dir, output_node_names):

    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export " "directory: %s" % model_dir
        )

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    absolute_model_dir = "/".join(input_checkpoint.split("/")[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"
    clear_devices = True
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + ".meta", clear_devices=clear_devices)
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), output_node_names.split(",")
        )
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


# In[13]:


freeze_graph("model-test", strings)


# In[14]:


with open("dictionary-test.json", "w") as fopen:
    fopen.write(json.dumps(dictionary))


# In[15]:


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


# In[16]:


g = load_graph("model-test/frozen_model.pb")


# In[19]:


x = g.get_tensor_by_name("import/Placeholder:0")
y = g.get_tensor_by_name("import/logits:0")
test_sess = tf.InteractiveSession(graph=g)
results = np.argmax(test_sess.run(y, feed_dict={x: batch_x}), axis=1)
results


# In[ ]:
