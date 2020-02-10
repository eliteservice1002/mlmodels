#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import threading
from queue import Queue

import requests
import tensorflow as tf

import model

# In[4]:


get_ipython().run_cell_magic(
    "time",
    "",
    'tf.reset_default_graph()\nsess = tf.InteractiveSession()\nmodel.Model(3, 256, 64, 2, 0.0001)\nsess.run(tf.global_variables_initializer())\nsaver = tf.train.Saver(tf.global_variables())\nsaver.restore(sess, os.getcwd() + "/model-rnn-vector-huber.ckpt")',
)


# In[5]:


get_ipython().run_cell_magic(
    "time",
    "",
    "tf.reset_default_graph()\ndef load_graph(frozen_graph_filename):\n    with tf.gfile.GFile(frozen_graph_filename, \"rb\") as f:\n        graph_def = tf.GraphDef()\n        graph_def.ParseFromString(f.read())\n    with tf.Graph().as_default() as graph:\n        tf.import_graph_def(graph_def)\n    return graph\n\ng=load_graph('frozen_model.pb')\nx = g.get_tensor_by_name('import/Placeholder:0')\ny = g.get_tensor_by_name('import/logits:0')\nsess = tf.InteractiveSession(graph=g)",
)


# ## Freeze model loaded more faster than dynamic model

# In[9]:


def run_parallel_in_threads(target, args_list):
    globalparas = []
    result = Queue()

    def task_wrapper(*args):
        result.put(target(*args))

    threads = [threading.Thread(target=task_wrapper, args=args) for args in args_list]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    while not result.empty():
        globalparas.append(result.get())
    globalparas = list(filter(None, globalparas))
    return globalparas


def get_time(text, type_api, i):
    response = str(requests.get("http://192.168.0.102:8033/%s?text=%s" % (type_api, text)).content)
    return [response, i]


# # Stress test 20 requests concurrently on dynamic graph

# In[27]:


CONCURRENT = 20
threads = []
for i in range(CONCURRENT):
    threads.append(("Freeze model loaded more faster than dynamic model", "dynamic", i))
outputs = run_parallel_in_threads(get_time, threads)
total = 0
for i in outputs:
    total += float(i[0][2:-1])
    print("thread %d, time taken %f s" % (i[1], float(i[0][2:-1])))

print("total time taken %f s, average time taken %f s" % (total, total / CONCURRENT))


# # Stress test 20 requests concurrently on static graph

# In[29]:


CONCURRENT = 20
threads = []
for i in range(CONCURRENT):
    threads.append(("Freeze model loaded more faster than dynamic model", "static", i))
outputs = run_parallel_in_threads(get_time, threads)
total = 0
for i in outputs:
    total += float(i[0][2:-1])
    print("thread %d, time taken %f s" % (i[1], float(i[0][2:-1])))

print("total time taken %f s, average time taken %f s" % (total, total / CONCURRENT))


# # Run 5 experiments on stress test 20 requests concurrently on dynamic graph

# In[34]:


total_experiments = 0
for _ in range(5):
    CONCURRENT = 20
    threads = []
    for i in range(CONCURRENT):
        threads.append(("Freeze model loaded more faster than dynamic model", "dynamic", i))
    outputs = run_parallel_in_threads(get_time, threads)
    total = 0
    for i in outputs:
        total += float(i[0][2:-1])
    total_experiments += total

print(
    "time taken to run experiments %f s, average %f s" % (total_experiments, total_experiments / 5)
)


# # Run 5 experiments on stress test 20 requests concurrently on static graph

# In[35]:


total_experiments = 0
for _ in range(5):
    CONCURRENT = 20
    threads = []
    for i in range(CONCURRENT):
        threads.append(("Freeze model loaded more faster than dynamic model", "static", i))
    outputs = run_parallel_in_threads(get_time, threads)
    total = 0
    for i in outputs:
        total += float(i[0][2:-1])
    total_experiments += total

print(
    "time taken to run experiments %f s, average %f s" % (total_experiments, total_experiments / 5)
)


# In[ ]:
