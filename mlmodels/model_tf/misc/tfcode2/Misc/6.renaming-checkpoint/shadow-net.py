#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

import numpy as np
import tensorflow as tf

import cv2
from crnn_model import crnn_model

# In[2]:


with open("crnn_model/char_dict.json") as fopen:
    char_dict = json.load(fopen)

with open("crnn_model/ord_map.json") as fopen:
    order_dict = json.load(fopen)


# In[3]:


class Model:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, (None, None, 3))
        image = tf.expand_dims(self.X, 0)
        image = tf.image.resize_images(image, [32, 100])
        num_classes = 37
        net = crnn_model.ShadowNet(
            phase="Test", hidden_nums=256, layers_nums=2, num_classes=num_classes
        )
        with tf.variable_scope("shadow"):
            net_out = net.build_shadownet(inputdata=image)
        self.decode, _ = tf.nn.ctc_beam_search_decoder(
            net_out, 25 * np.ones(1), merge_repeated=False
        )


# In[4]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, "rename-checkpoint2/model.ckpt")


# In[5]:


image = cv2.imread("back-car.jpeg", cv2.IMREAD_COLOR)


# In[6]:


image.shape


# In[7]:


output = sess.run(model.decode, feed_dict={model.X: image})[0]


# In[8]:


output


# In[9]:


def sparse_tensor_to_str(sparse_tensor):
    indices = sparse_tensor.indices
    values = sparse_tensor.values
    values = np.array([order_dict[str(tmp)] for tmp in values])
    dense_shape = sparse_tensor.dense_shape

    number_lists = np.ones(dense_shape, dtype=values.dtype)
    str_lists = []
    res = []
    for i, index in enumerate(indices):
        number_lists[index[0], index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append([char_dict[val] for val in number_list])
    for str_list in str_lists:
        res.append("".join(c for c in str_list if c != "\x00"))
    return res


# In[10]:


sparse_tensor_to_str(output)[0]


# In[11]:


char_list = "0123456789abcdefghijklmnopqrstuvwxyz "


# In[12]:


def sparse_tensor_to_str2(spares_tensor):

    indices = spares_tensor.indices
    values = spares_tensor.values
    dense_shape = spares_tensor.dense_shape

    number_lists = np.ones(dense_shape, dtype=values.dtype)
    str_lists = []
    res = []
    for i, index in enumerate(indices):
        number_lists[index[0], index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append([char_list[val] for val in number_list])
    for str_list in str_lists:
        res.append("".join(c for c in str_list if c != "1"))
    return res


# In[13]:


sparse_tensor_to_str2(output)


# In[ ]:
