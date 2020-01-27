#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

# In[2]:


mnist = input_data.read_data_sets("", one_hot=True)


# In[3]:


class Model:
    def __init__(self, learning_rate, y_shape):
        self.X = tf.placeholder(tf.float32, (None, 28, 28, 1))
        self.Y = tf.placeholder(tf.float32, (None, y_shape))

        def convolutionize(x, conv_w, h=1):
            return tf.nn.conv2d(input=x, filter=conv_w, strides=[1, h, h, 1], padding="SAME")

        def pooling(wx):
            return tf.nn.max_pool(wx, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        w1 = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.5))
        b1 = tf.Variable(tf.zeros(shape=[16]))
        w2 = tf.Variable(tf.random_normal([3, 3, 16, 8], stddev=0.5))
        b2 = tf.Variable(tf.zeros(shape=[8]))
        w3 = tf.Variable(tf.random_normal([3, 3, 8, 8], stddev=0.5))
        b3 = tf.Variable(tf.zeros(shape=[8]))
        w4 = tf.Variable(tf.random_normal([128, y_shape], stddev=0.5))
        b4 = tf.Variable(tf.zeros(shape=[y_shape]))

        conv1 = pooling(tf.nn.sigmoid(convolutionize(self.X, w1) + b1))
        conv2 = pooling(tf.nn.sigmoid(convolutionize(conv1, w2) + b2))
        conv3 = pooling(tf.nn.sigmoid(convolutionize(conv2, w3) + b3))
        conv3 = tf.reshape(conv3, [-1, 128])
        self.logits = tf.matmul(conv3, w4) + b4

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))


# In[4]:


learning_rate = 0.01
sess = tf.InteractiveSession()
model = Model(learning_rate, mnist.train.labels.shape[1])
sess.run(tf.global_variables_initializer())


# In[6]:


BATCH_SIZE = 128

for i in range(10):
    EPOCH.append(i)
    TOTAL_LOSS, ACCURACY = 0, 0
    for n in range(0, (mnist.train.images.shape[0] // BATCH_SIZE) * BATCH_SIZE, BATCH_SIZE):
        batch_x = mnist.train.images[n : n + BATCH_SIZE, :].reshape((-1, 28, 28, 1))
        cost, _ = sess.run(
            [model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: mnist.train.labels[n : n + BATCH_SIZE, :]},
        )
        ACCURACY += sess.run(
            model.accuracy,
            feed_dict={model.X: batch_x, model.Y: mnist.train.labels[n : n + BATCH_SIZE, :]},
        )
        TOTAL_LOSS += cost

    TOTAL_LOSS /= mnist.train.images.shape[0] // BATCH_SIZE
    ACCURACY /= mnist.train.images.shape[0] // BATCH_SIZE
    LOSS.append(TOTAL_LOSS)
    ACC.append(ACCURACY)
    print("epoch: %d, avg loss %f, avg acc %f" % (i + 1, TOTAL_LOSS, ACCURACY))


# In[7]:


model_version = 1
export_model_dir = "./serving/versions"


# In[8]:


export_path_base = export_model_dir
export_path = os.path.join(
    tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(str(model_version))
)
print("Exporting trained model to", export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_input = tf.saved_model.utils.build_tensor_info(model.X)
tensor_output = tf.saved_model.utils.build_tensor_info(model.logits)

prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs={"image": tensor_input},
    outputs={"logits": tensor_output},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
)

builder.add_meta_graph_and_variables(
    sess,
    [tf.saved_model.tag_constants.SERVING],
    signature_def_map={"predict_classes": prediction_signature},
)

builder.save(as_text=True)


# Open a new terminal, and run,
# ```bash
# tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=serving/versions
# ```

# In[9]:


# In[13]:


server = "localhost:9000"
host, port = server.split(":")
img = mnist.test.images[0].reshape((28, 28, 1))
np.expand_dims(img, 0).shape


# In[12]:


channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)


# In[14]:


request = predict_pb2.PredictRequest()
request.model_spec.name = "mnist"
request.model_spec.signature_name = "predict_classes"
request.inputs["image"].CopyFrom(
    tf.contrib.util.make_tensor_proto(
        np.expand_dims(img, 0).astype(dtype=np.float32), shape=[1, 28, 28, 1]
    )
)


# In[15]:


result_future = stub.Predict(request, 30.0)


# In[17]:


np.argmax(result_future.outputs["logits"].float_val)


# In[19]:


np.argmax(mnist.test.labels[0])


# In[ ]:
