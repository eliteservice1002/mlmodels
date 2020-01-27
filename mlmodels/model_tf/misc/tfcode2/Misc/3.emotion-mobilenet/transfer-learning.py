#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from random import shuffle

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.misc import imread, imresize
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import mobilenet_v2

tf.__version__


# Make sure you download this data and extract in the same directory,
#
# https://drive.google.com/open?id=1V9fy_Me9ZjmMTJoTWz0L8AUdIW5k35bE

# In[2]:


checkpoint_name = "mobilenet_v2_1.0_224"
url = "https://storage.googleapis.com/mobilenet_v2/checkpoints/" + checkpoint_name + ".tgz"
print("Downloading from ", url)
get_ipython().system("wget {url}")
print("Unpacking")
get_ipython().system("tar -xvf {checkpoint_name}.tgz")
checkpoint = checkpoint_name + ".ckpt"


# In[3]:


batch_size = 32
epoch = 10
learning_rate = 1e-3
data_location = "Crop/"


# In[4]:


img_lists = os.listdir(data_location)
shuffle(img_lists)
img_labels = [i.split("--")[0] for i in img_lists]
img_Y = LabelEncoder().fit_transform(img_labels)
img_lists = [data_location + i for i in img_lists]


# In[5]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32, [None, 224, 224, 1])
Y = tf.placeholder(tf.int32, [None])
images = tf.image.grayscale_to_rgb(X)
images = images / 128.0 - 1
with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
    logits, endpoints = mobilenet_v2.mobilenet(images)
logits = tf.nn.relu6(logits)
emotion_logits = slim.fully_connected(
    logits,
    7,
    activation_fn=None,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    weights_regularizer=slim.l2_regularizer(1e-5),
    scope="emo/emotion_1",
    reuse=False,
)
emotion_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=Y, logits=emotion_logits
)
emotion_cross_entropy_mean = tf.reduce_mean(emotion_cross_entropy)
cost = tf.add_n(
    [emotion_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
)
emotion_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(emotion_logits, Y, 1), tf.float32))
global_step = tf.Variable(0, name="global_step", trainable=False)
# only train on our emotion layers
emotion_vars = [var for var in tf.trainable_variables() if var.name.find("emotion_") >= 0]
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=emotion_vars)

sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MobilenetV2")
saver = tf.train.Saver(var_list=var_lists)
saver.restore(sess, checkpoint)
saver = tf.train.Saver(tf.global_variables())
# test save
saver.save(sess, "new/emotion-checkpoint-mobilenet.ckpt")


# In[6]:


batching = (len(img_lists) // batch_size) * batch_size
for i in range(epoch):
    total_loss, total_acc = 0, 0
    for k in tqdm(range(0, batching, batch_size), desc="minibatch loop"):
        batch_x = np.zeros((batch_size, 224, 224, 1))
        for n in range(batch_size):
            img = imresize(imread(img_lists[k + n]), (224, 224))
            batch_x[n, :, :, 0] = img
        loss, acc, _ = sess.run(
            [cost, emotion_accuracy, optimizer],
            feed_dict={X: batch_x, Y: img_Y[k : k + batch_size]},
        )
        total_loss += loss
        total_acc += acc
    total_loss /= len(img_lists) // batch_size
    total_acc /= len(img_lists) // batch_size
    print("epoch: %d, avg loss: %f, avg accuracy: %f" % (i + 1, total_loss, total_acc))
    saver.save(sess, "new/emotion-checkpoint-mobilenet.ckpt")


# In[21]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32, [None, 224, 224, 1])
Y = tf.placeholder(tf.int32, [None])
images = tf.image.grayscale_to_rgb(X)
images = images / 128.0 - 1
with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    logits, endpoints = mobilenet_v2.mobilenet(images)
logits = tf.nn.relu6(logits)
emotion_logits = slim.fully_connected(
    logits,
    7,
    activation_fn=None,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    weights_regularizer=slim.l2_regularizer(1e-5),
    scope="emo/emotion_1",
    reuse=False,
)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, "new/emotion-checkpoint-mobilenet.ckpt")


# In[22]:


batching = (len(img_lists) // batch_size) * batch_size
results = []
for k in tqdm(range(0, batching, batch_size), desc="minibatch loop"):
    batch_x = np.zeros((batch_size, 224, 224, 1))
    for n in range(batch_size):
        img = imresize(imread(img_lists[k + n]), (224, 224))
        batch_x[n, :, :, 0] = img
    results += sess.run(tf.argmax(emotion_logits, 1), feed_dict={X: batch_x}).tolist()


# In[23]:


print(metrics.classification_report(img_Y[:batching], results, target_names=np.unique(img_labels)))


# In[ ]:
