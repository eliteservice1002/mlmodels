#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system("pip3 install tqdm requests dill")


# In[2]:


import os

import requests
import tensorflow as tf
from tqdm import tqdm

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from sparkflow.graph_utils import build_adam_config, build_graph
from sparkflow.tensorflow_async import SparkAsyncDL


def download_from_url(url, dst):
    file_size = int(requests.head(url).headers["Content-Length"])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte, unit="B", unit_scale=True, desc=url.split("/")[-1]
    )
    req = requests.get(url, headers=header, stream=True)
    with (open(dst, "ab")) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


# In[3]:


download_from_url(
    "https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/mnist_train.csv",
    "mnist_train.csv",
)


# In[4]:


def cnn_model():
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    fc1 = tf.contrib.layers.flatten(conv2)
    out = tf.layers.dense(fc1, 10)
    z = tf.argmax(out, 1, name="out")
    loss = tf.losses.softmax_cross_entropy(y, out)
    return loss


# In[5]:


sparkSession = SparkSession.builder.appName("csv").getOrCreate()


# In[6]:


df = sparkSession.read.csv("mnist_train.csv", header=True, inferSchema=True)


# In[7]:


va = VectorAssembler(inputCols=df.columns[1:785], outputCol="features").transform(df)


# In[8]:


va.select("label").show(1)


# In[9]:


encoded = (
    OneHotEncoder(inputCol="label", outputCol="labels", dropLast=False)
    .transform(va)
    .select(["features", "labels"])
)


# In[10]:


mg = build_graph(cnn_model)
adam_config = build_adam_config(learning_rate=0.001, beta1=0.9, beta2=0.999)


# In[11]:


spark_model = SparkAsyncDL(
    inputCol="features",
    tensorflowGraph=mg,
    tfInput="x:0",
    tfLabel="y:0",
    tfOutput="out:0",
    tfOptimizer="adam",
    miniBatchSize=300,
    miniStochasticIters=1,
    shufflePerIter=True,
    iters=50,
    predictionCol="predicted",
    labelCol="labels",
    partitions=3,
    verbose=1,
    optimizerOptions=adam_config,
)


# In[13]:


fitted_model = spark_model.fit(encoded)


# In[14]:


predictions = fitted_model.transform(encoded)


# In[15]:


predictions.show(1)


# In[ ]:


evaluator = MulticlassClassificationEvaluator(
    labelCol="labels", predictionCol="predicted", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))


# In[ ]:
