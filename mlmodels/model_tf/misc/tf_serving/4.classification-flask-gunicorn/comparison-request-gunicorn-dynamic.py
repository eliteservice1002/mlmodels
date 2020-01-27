#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import threading
from queue import Queue

import requests

# In[2]:


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


# # Stress test 50 requests concurrently on dynamic graph, when worker = 1

# #### Please run,
#
# ```bash
# bash run-gunicorn-dynamic.sh 1
# ```

# In[10]:


CONCURRENT = 50
threads = []
for i in range(CONCURRENT):
    threads.append(("Freeze model loaded more faster than dynamic model", "dynamic", i))
outputs = run_parallel_in_threads(get_time, threads)
total = 0
for i in outputs:
    total += float(i[0][2:-1])
    print("thread %d, time taken %f s" % (i[1], float(i[0][2:-1])))

print("total time taken %f s, average time taken %f s" % (total, total / CONCURRENT))


# # Stress test 50 requests concurrently on dynamic graph, when worker = 2

# #### Please run,
#
# ```bash
# bash run-gunicorn-dynamic.sh 2
# ```

# In[12]:


CONCURRENT = 50
threads = []
for i in range(CONCURRENT):
    threads.append(("Freeze model loaded more faster than dynamic model", "dynamic", i))
outputs = run_parallel_in_threads(get_time, threads)
total = 0
for i in outputs:
    total += float(i[0][2:-1])
    print("thread %d, time taken %f s" % (i[1], float(i[0][2:-1])))

print("total time taken %f s, average time taken %f s" % (total, total / CONCURRENT))


# # Stress test 50 requests concurrently on dynamic graph, when worker = 5

# #### Please run,
#
# ```bash
# bash run-gunicorn-dynamic.sh 5
# ```

# In[13]:


CONCURRENT = 50
threads = []
for i in range(CONCURRENT):
    threads.append(("Freeze model loaded more faster than dynamic model", "dynamic", i))
outputs = run_parallel_in_threads(get_time, threads)
total = 0
for i in outputs:
    total += float(i[0][2:-1])
    print("thread %d, time taken %f s" % (i[1], float(i[0][2:-1])))

print("total time taken %f s, average time taken %f s" % (total, total / CONCURRENT))


# # Stress test 50 requests concurrently on dynamic graph, when worker = 7

# #### Please run,
#
# ```bash
# bash run-gunicorn-dynamic.sh 7
# ```

# In[14]:


CONCURRENT = 50
threads = []
for i in range(CONCURRENT):
    threads.append(("Freeze model loaded more faster than dynamic model", "dynamic", i))
outputs = run_parallel_in_threads(get_time, threads)
total = 0
for i in outputs:
    total += float(i[0][2:-1])
    print("thread %d, time taken %f s" % (i[1], float(i[0][2:-1])))

print("total time taken %f s, average time taken %f s" % (total, total / CONCURRENT))


# # Stress test 50 requests concurrently on dynamic graph, when worker = 10

# #### Please run,
#
# ```bash
# bash run-gunicorn-dynamic.sh 10
# ```

# In[15]:


CONCURRENT = 50
threads = []
for i in range(CONCURRENT):
    threads.append(("Freeze model loaded more faster than dynamic model", "dynamic", i))
outputs = run_parallel_in_threads(get_time, threads)
total = 0
for i in outputs:
    total += float(i[0][2:-1])
    print("thread %d, time taken %f s" % (i[1], float(i[0][2:-1])))

print("total time taken %f s, average time taken %f s" % (total, total / CONCURRENT))


# In[ ]:
