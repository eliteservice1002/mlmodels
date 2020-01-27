#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import tensorflow as tf
from scipy.io.wavfile import write
from tqdm import tqdm

from utils import *

# In[2]:


def pad_causal(x, size, rate):
    pad_len = (size - 1) * rate
    return tf.pad(x, [[0, 0], [pad_len, 0], [0, 0]])


class Wavenet:
    def __init__(self, num_layers, size_layers, num_blocks=3, block_size=128, dropout=1.0):
        self.X = tf.placeholder(tf.int32, (None, None))
        lookup_table = tf.get_variable(
            "lookup_table",
            dtype=tf.float32,
            shape=[len(vocab), size_layers],
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        )
        lookup_table = tf.concat((tf.zeros(shape=[1, size_layers]), lookup_table[1:, :]), 0)
        forward = tf.nn.embedding_lookup(lookup_table, self.X)
        self.Y = tf.placeholder(tf.float32, (None, None, n_mels * resampled))
        self.decoder_inputs = tf.concat((tf.zeros_like(self.Y[:, :1, :]), self.Y[:, :-1, :]), 1)
        self.decoder_inputs = self.decoder_inputs[:, :, -n_mels:]
        print(self.decoder_inputs)
        self.Z = tf.placeholder(tf.float32, (None, None, fourier_window_size // 2 + 1))

        def residual_block(x, size, rate, block):
            with tf.variable_scope("block_%d_%d" % (block, rate), reuse=False):
                conv_filter = tf.layers.conv1d(
                    x,
                    x.shape[2] // 4,
                    kernel_size=size,
                    strides=1,
                    padding="same",
                    dilation_rate=rate,
                    activation=tf.nn.tanh,
                )
                conv_gate = tf.layers.conv1d(
                    x,
                    x.shape[2] // 4,
                    kernel_size=size,
                    strides=1,
                    padding="same",
                    dilation_rate=rate,
                    activation=tf.nn.sigmoid,
                )
                out = tf.multiply(conv_filter, conv_gate)
                out = tf.layers.conv1d(
                    out, block_size, kernel_size=1, strides=1, padding="same", activation=tf.nn.tanh
                )
                return tf.add(x, out), out

        with tf.variable_scope("encode", reuse=False):
            forward = tf.layers.conv1d(
                forward, block_size, kernel_size=1, strides=1, padding="SAME"
            )
            zeros = tf.zeros_like(forward)
            for i in range(num_blocks):
                for r in [1, 2, 4, 8, 16]:
                    forward, s = residual_block(forward, size=7, rate=r, block=i)
                    zeros = tf.add(zeros, s)
            forward = tf.layers.conv1d(
                forward, block_size, kernel_size=1, strides=1, padding="SAME"
            )
            encoded = tf.layers.conv1d(forward, n_mels, kernel_size=1, strides=1, padding="SAME")
            encoded = tf.reduce_mean(encoded, axis=1, keepdims=True)

        with tf.variable_scope("y_hat", reuse=False):
            encoded_tile = tf.tile(encoded, [1, tf.shape(self.decoder_inputs)[1], 1])
            self.decoder_inputs = tf.multiply(self.decoder_inputs, encoded_tile)
            forward = tf.layers.conv1d(
                self.decoder_inputs, block_size, kernel_size=1, strides=1, padding="SAME"
            )
            zeros = tf.zeros_like(forward)
            for i in range(num_blocks):
                for r in [1, 2, 4, 8, 16]:
                    forward, s = residual_block(forward, size=7, rate=r, block=i)
                    zeros = tf.add(zeros, s)
            forward = tf.layers.conv1d(
                forward, block_size, kernel_size=1, strides=1, padding="SAME"
            )
            self.Y_hat = tf.layers.conv1d(
                forward, n_mels * resampled, kernel_size=1, strides=1, padding="SAME"
            )

        with tf.variable_scope("z_hat", reuse=False):
            forward = tf.reshape(self.Y_hat, [tf.shape(self.Y_hat)[0], -1, n_mels])
            forward = tf.layers.conv1d(
                forward, size_layers, kernel_size=1, strides=1, padding="SAME"
            )
            zeros = tf.zeros_like(forward)
            for i in range(num_blocks):
                for r in [1, 2, 4, 8, 16]:
                    forward, s = residual_block(forward, size=7, rate=r, block=i)
                    zeros = tf.add(zeros, s)
            forward = tf.layers.conv1d(
                forward, block_size, kernel_size=1, strides=1, padding="SAME"
            )
            self.Z_hat = tf.layers.conv1d(
                forward, 1 + fourier_window_size // 2, kernel_size=1, strides=1, padding="SAME"
            )

        self.loss1 = tf.reduce_mean(tf.abs(self.Y_hat - self.Y))
        self.loss2 = tf.reduce_mean(tf.abs(self.Z_hat - self.Z))
        self.loss = self.loss1 + self.loss2
        self.lr = 1e-3
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


# In[3]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

size_layers = 128
learning_rate = 1e-3
num_layers = 2

model = Wavenet(num_layers, size_layers)
sess.run(tf.global_variables_initializer())


# In[4]:


paths, lengths, texts, raw_texts = [], [], [], []
text_files = [f for f in os.listdir("mel") if f.endswith(".npy")]
for fpath in text_files:
    with open("%s/%s" % (path, fpath.replace("npy", "txt"))) as fopen:
        text = fopen.read()
    paths.append(fpath.replace(".npy", ""))
    text = text_normalize(text)
    raw_texts.append(text)
    text = text + "E"
    texts.append(np.array([char2idx[char] for char in text], np.int32))
    lengths.append(len(text))


# In[5]:


paths[:2], lengths[:2], texts[:2]


# In[6]:


def dynamic_batching(paths):
    files, max_y, max_z = [], 0, 0
    for n in range(len(paths)):
        files.append(get_cached(paths[n]))
        if files[-1][0].shape[0] > max_y:
            max_y = files[-1][0].shape[0]
        if files[-1][1].shape[0] > max_z:
            max_z = files[-1][1].shape[0]
    return files, max_y, max_z


# In[7]:


EPOCH = 30
for i in range(EPOCH):
    pbar = tqdm(range(0, len(paths), batch_size), desc="minibatch loop")
    for k in pbar:
        index = min(k + batch_size, len(paths))
        files, max_y, max_z = dynamic_batching(paths[k:index])
        max_x = max(lengths[k:index])
        batch_x = np.zeros((batch_size, max_x))
        batch_y = np.zeros((batch_size, max_y, n_mels * resampled))
        batch_z = np.zeros((batch_size, max_z, fourier_window_size // 2 + 1))
        for n in range(len(files)):
            batch_x[n, :] = np.pad(
                texts[k + n], ((0, max_x - texts[k + n].shape[0])), mode="constant"
            )
            batch_y[n, :, :] = np.pad(
                files[n][0], ((0, max_y - files[n][0].shape[0]), (0, 0)), mode="constant"
            )
            batch_z[n, :, :] = np.pad(
                files[n][1], ((0, max_z - files[n][1].shape[0]), (0, 0)), mode="constant"
            )
        _, cost = sess.run(
            [model.optimizer, model.loss],
            feed_dict={model.X: batch_x, model.Y: batch_y, model.Z: batch_z},
        )
        pbar.set_postfix(cost=cost)


# In[8]:


y_hat = np.zeros((1, 50, n_mels * resampled), np.float32)
for j in tqdm(range(50)):
    _y_hat = sess.run(model.Y_hat, {model.X: [texts[0]], model.Y: y_hat})
    y_hat[:, j, :] = _y_hat[:, j, :]


# In[9]:


mags = sess.run(model.Z_hat, {model.Y_hat: y_hat})


# In[10]:


audio = spectrogram2wav(mags[0])


# In[11]:


print("saving: %s" % (raw_texts[0]))
write(os.path.join("test.wav"), sample_rate, audio)


# In[ ]:
