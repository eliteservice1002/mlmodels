#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.misc import imread, imresize

import inception_v1

# just remove line below if want to use GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


# In[2]:


with open("real-label.json", "r") as fopen:
    labels = json.load(fopen)


# In[3]:


img = imread("fucking-panda.jpg")
img.shape


# In[4]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32, [None, None, 3])
image = X / 128.0 - 1
image = tf.expand_dims(image, 0)
image = tf.image.resize_images(image, (224, 224))
with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
    logits, endpoints = inception_v1.inception_v1(image, num_classes=1001, is_training=False)
sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV1")
saver = tf.train.Saver(var_list=var_lists)
saver.restore(sess, "inception_v1.ckpt")


# In[5]:


get_ipython().run_cell_magic(
    "time", "", "sess.run(logits,feed_dict={X:img})\n# first time slow, GPU caching"
)


# In[8]:


get_ipython().run_cell_magic(
    "time",
    "",
    "labels[str(np.argmax(sess.run(logits,feed_dict={X:img})[0]))]\n# repeat same experiment to get accurate time",
)


# In[ ]:
