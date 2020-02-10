#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system("pip3 install tqdm requests dill")


# In[1]:


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


# In[2]:


download_from_url(
    "https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/mnist_train.csv",
    "mnist_train.csv",
)


# In[3]:


def small_model():
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
    layer1 = tf.layers.dense(x, 256, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 256, activation=tf.nn.relu)
    out = tf.layers.dense(layer2, 10)
    z = tf.argmax(out, 1, name="out")
    loss = tf.losses.softmax_cross_entropy(y, out)
    return loss


# In[4]:


sparkSession = SparkSession.builder.appName("csv").getOrCreate()


# In[5]:


df = sparkSession.read.csv("mnist_train.csv", header=True, inferSchema=True)


# In[6]:


va = VectorAssembler(inputCols=df.columns[1:785], outputCol="features").transform(df)


# In[7]:


va.select("label").show(1)


# In[8]:


encoded = (
    OneHotEncoder(inputCol="label", outputCol="labels", dropLast=False)
    .transform(va)
    .select(["features", "labels"])
)


# In[9]:


mg = build_graph(small_model)
adam_config = build_adam_config(learning_rate=0.001, beta1=0.9, beta2=0.999)


# In[10]:


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


# In[12]:


fitted_model = spark_model.fit(encoded)


# In[13]:


predictions = fitted_model.transform(encoded)


# In[14]:


predictions.show(1)


# In[ ]:


evaluator = MulticlassClassificationEvaluator(
    labelCol="labels", predictionCol="predicted", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))


# In[ ]:
