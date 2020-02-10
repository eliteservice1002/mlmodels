#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

sns.set()


# In[2]:


mcycle = pd.read_csv("mcycle.txt", delimiter="\t")
mcycle.times = (mcycle.times - mcycle.times.mean()) / mcycle.times.std()
mcycle.accel = (mcycle.accel - mcycle.accel.mean()) / mcycle.accel.std()

times = np.expand_dims(mcycle.times.values, 1)
accel = np.expand_dims(mcycle.accel.values, 1)


# In[3]:


class q_model:
    def __init__(self, sess, quantiles, in_shape=1, out_shape=1, batch_size=32):

        self.sess = sess

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.batch_size = batch_size

        self.outputs = []
        self.losses = []
        self.loss_history = []

        self.build_model()

    def build_model(self, scope="q_model", reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse) as scope:
            self.x = tf.placeholder(tf.float32, shape=(None, self.in_shape))
            self.y = tf.placeholder(tf.float32, shape=(None, self.out_shape))

            self.layer0 = tf.layers.dense(self.x, units=32, activation=tf.nn.relu)
            self.layer1 = tf.layers.dense(self.layer0, units=32, activation=tf.nn.relu)

            for i in range(self.num_quantiles):
                q = self.quantiles[i]

                output = tf.layers.dense(self.layer1, 1, name="{}_q{}".format(i, int(q * 100)))
                self.outputs.append(output)

                error = tf.subtract(self.y, output)
                loss = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1)

                self.losses.append(loss)

            self.combined_loss = tf.reduce_mean(tf.add_n(self.losses))
            self.train_step = tf.train.AdamOptimizer().minimize(self.combined_loss)

    def fit(self, x, y, epochs=100):
        for epoch in range(epochs):
            epoch_losses = []
            for idx in range(0, x.shape[0], self.batch_size):
                batch_x = x[idx : min(idx + self.batch_size, x.shape[0]), :]
                batch_y = y[idx : min(idx + self.batch_size, y.shape[0]), :]

                feed_dict = {self.x: batch_x, self.y: batch_y}

                _, c_loss = self.sess.run([self.train_step, self.combined_loss], feed_dict)
                epoch_losses.append(c_loss)

            epoch_loss = np.mean(epoch_losses)
            self.loss_history.append(epoch_loss)
            if epoch % 100 == 0:
                print("Epoch {}: {}".format(epoch, epoch_loss))

    def predict(self, x):
        feed_dict = {self.x: x}
        predictions = sess.run(self.outputs, feed_dict)

        return predictions


# In[4]:


quantiles = [0.1, 0.5, 0.9]
sess = tf.InteractiveSession()
model = q_model(sess, quantiles, batch_size=32)
init_op = tf.global_variables_initializer()
sess.run(init_op)


# In[5]:


model.fit(times, accel, 2000)


# In[7]:


plt.figure(figsize=(10, 5))

test_times = np.expand_dims(np.linspace(times.min(), times.max(), 200), 1)
predictions = model.predict(test_times)

plt.scatter(times, accel)
for i, prediction in enumerate(predictions):
    plt.plot(test_times, prediction, label="{}th Quantile".format(int(model.quantiles[i] * 100)))

plt.legend()
plt.show()
