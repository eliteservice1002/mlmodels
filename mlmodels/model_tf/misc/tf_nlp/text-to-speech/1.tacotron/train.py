#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import tensorflow as tf
from scipy.io.wavfile import write
from tqdm import tqdm

from tacotron import Tacotron
from utils import *

# In[2]:


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


# In[3]:


paths[:2], lengths[:2], texts[:2]


# In[4]:


def dynamic_batching(paths):
    files, max_y, max_z = [], 0, 0
    for n in range(len(paths)):
        files.append(get_cached(paths[n]))
        if files[-1][0].shape[0] > max_y:
            max_y = files[-1][0].shape[0]
        if files[-1][1].shape[0] > max_z:
            max_z = files[-1][1].shape[0]
    return files, max_y, max_z


# In[5]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Tacotron()
sess.run(tf.global_variables_initializer())


# In[6]:


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
    align = sess.run(model.alignments, feed_dict={model.X: batch_x, model.Y: batch_y})
    plot_alignment(align[0, :, :])


# In[7]:


y_hat = np.zeros((1, 200, n_mels * resampled), np.float32)
for j in tqdm(range(200)):
    _y_hat = sess.run(model.Y_hat, {model.X: [texts[0]], model.Y: y_hat})
    y_hat[:, j, :] = _y_hat[:, j, :]


# In[8]:


mags = sess.run(model.Z_hat, {model.Y_hat: y_hat})


# In[9]:


audio = spectrogram2wav(mags[0])


# In[10]:


print("saving: %s" % (raw_texts[0]))
write(os.path.join("test.wav"), sample_rate, audio)


# In[ ]:
