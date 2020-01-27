#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
import pickle
import random
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

# In[2]:


maxlen = 50
location = os.getcwd()
num_layers = 3
size_layer = 256
learning_rate = 0.0001
batch = 100


# In[3]:


with open("dataset-sentiment.p", "rb") as fopen:
    df = pickle.load(fopen)
with open("vector-sentiment.p", "rb") as fopen:
    vectors = pickle.load(fopen)
with open("dictionary-sentiment.p", "rb") as fopen:
    dictionary = pickle.load(fopen)
label = np.unique(df[:, 1])


# In[4]:


train_X, test_X, train_Y, test_Y = train_test_split(df[:, 0], df[:, 1].astype("int"), test_size=0.2)


# In[5]:


class Model:
    def __init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate):
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(size_layer)

        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        self.X = tf.placeholder(tf.float32, [None, None, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        drop = tf.contrib.rnn.DropoutWrapper(self.rnn_cells, output_keep_prob=0.5)
        self.outputs, self.last_state = tf.nn.dynamic_rnn(drop, self.X, dtype=tf.float32)
        self.rnn_W = tf.Variable(tf.random_normal((size_layer, dimension_output)))
        self.rnn_B = tf.Variable(tf.random_normal([dimension_output]))
        # put 'logits' name is very important
        self.logits = tf.add(tf.matmul(self.outputs[:, -1], self.rnn_W), self.rnn_B, name="logits")
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        l2 = sum(0.0005 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.cost += l2
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[6]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(num_layers, size_layer, vectors.shape[1], label.shape[0], learning_rate)
sess.run(tf.global_variables_initializer())
dimension = vectors.shape[1]
saver = tf.train.Saver(tf.global_variables())
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 10, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:", EPOCH)
        break
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (train_X.shape[0] // batch) * batch, batch):
        batch_x = np.zeros((batch, maxlen, dimension))
        batch_y = np.zeros((batch, len(label)))
        for k in range(batch):
            tokens = train_X[i + k].split()[:maxlen]
            emb_data = np.zeros((maxlen, dimension), dtype=np.float32)
            for no, text in enumerate(tokens[::-1]):
                try:
                    emb_data[-1 - no, :] += vectors[dictionary[text], :]
                except Exception as e:
                    print(e)
                    continue
            batch_y[k, int(train_Y[i + k])] = 1.0
            batch_x[k, :, :] = emb_data[:, :]
        loss, _ = sess.run(
            [model.cost, model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        train_loss += loss
        train_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})

    for i in range(0, (test_X.shape[0] // batch) * batch, batch):
        batch_x = np.zeros((batch, maxlen, dimension))
        batch_y = np.zeros((batch, len(label)))
        for k in range(batch):
            tokens = test_X[i + k].split()[:maxlen]
            emb_data = np.zeros((maxlen, dimension), dtype=np.float32)
            for no, text in enumerate(tokens[::-1]):
                try:
                    emb_data[-1 - no, :] += vectors[dictionary[text], :]
                except:
                    continue
            batch_y[k, int(test_Y[i + k])] = 1.0
            batch_x[k, :, :] = emb_data[:, :]
        loss, acc = sess.run(
            [model.cost, model.accuracy], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        test_loss += loss
        test_acc += acc

    train_loss /= train_X.shape[0] // batch
    train_acc /= train_X.shape[0] // batch
    test_loss /= test_X.shape[0] // batch
    test_acc /= test_X.shape[0] // batch
    if test_acc > CURRENT_ACC:
        print("epoch:", EPOCH, ", pass acc:", CURRENT_ACC, ", current acc:", test_acc)
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
        saver.save(sess, os.getcwd() + "/model-rnn-vector-huber.ckpt")
    else:
        CURRENT_CHECKPOINT += 1
    print("time taken:", time.time() - lasttime)
    print(
        "epoch:",
        EPOCH,
        ", training loss:",
        train_loss,
        ", training acc:",
        train_acc,
        ", valid loss:",
        test_loss,
        ", valid acc:",
        test_acc,
    )
    EPOCH += 1


# In[7]:


# only load Variables, placeholder for input, and our logits
strings = ",".join(
    [
        n.name
        for n in tf.get_default_graph().as_graph_def().node
        if "Variable" in n.op or n.name.find("Placeholder") >= 0 or n.name.find("logits") == 0
    ]
)


# In[8]:


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


# In[9]:


freeze_graph("", strings)


# In[10]:


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


# In[11]:


g = load_graph("frozen_model.pb")


# In[12]:


for op in g.get_operations():
    print(op.name)


# In[15]:


x = g.get_tensor_by_name("import/Placeholder:0")
y = g.get_tensor_by_name("import/logits:0")
test_sess = tf.InteractiveSession(graph=g)
results = np.argmax(test_sess.run(tf.nn.softmax(y), feed_dict={x: batch_x}), axis=1)


# In[ ]:
