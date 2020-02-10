#!/usr/bin/env python
# coding: utf-8

# In[10]:


import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import cv2
import inception_v3

# just remove line below if want to use GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


# In[2]:


with open("real-label.json", "r") as fopen:
    labels = json.load(fopen)


# In[12]:


image = cv2.cvtColor(cv2.imread("husein.png"), cv2.COLOR_BGR2RGB)
if image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
image.shape


# In[16]:


plt.imshow(image)
plt.show()


# In[4]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32, [None, None, 3])
image = X / 128.0 - 1
image = tf.expand_dims(image, 0)
image = tf.image.resize_images(image, (299, 299))
with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits, endpoints = inception_v3.inception_v3(image, num_classes=1001, is_training=False)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "inception_v3.ckpt")


# In[5]:


get_ipython().run_cell_magic(
    "time", "", "sess.run(logits,feed_dict={X:img})\n# first time slow, GPU caching"
)


# In[14]:


get_ipython().run_cell_magic(
    "time",
    "",
    "labels[str(np.argmax(sess.run(logits,feed_dict={X:image})[0]))]\n# repeat same experiment to get accurate time",
)


# In[ ]:
