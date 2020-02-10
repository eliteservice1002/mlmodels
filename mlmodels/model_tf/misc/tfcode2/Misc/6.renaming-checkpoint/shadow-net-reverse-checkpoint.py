#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

import numpy as np
import tensorflow as tf

from crnn_model import crnn_model

# In[2]:


tf.reset_default_graph()
sess = tf.InteractiveSession()


# In[3]:


new_vars = []
vars_checkpoint = tf.contrib.framework.list_variables(
    "shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999"
)
for name, shape in vars_checkpoint:
    v = tf.contrib.framework.load_variable(
        "shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999", name
    )
    print(name, name.replace("BatchNorm", "batch_normalization"))
    new_vars.append(tf.Variable(v, name=name.replace("BatchNorm", "batch_normalization")))


# In[4]:


saver = tf.train.Saver(new_vars)
sess.run(tf.global_variables_initializer())
saver.save(sess, "rename-checkpoint2/model.ckpt")


# In[ ]:
