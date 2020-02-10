#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import json
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from tqdm import tqdm

# In[2]:


with open("ctexts.json", "r") as fopen:
    ctexts = json.load(fopen)

with open("headlines.json", "r") as fopen:
    headlines = json.load(fopen)


# In[3]:


def topic_modelling(string, n=500):
    vectorizer = TfidfVectorizer()
    tf = vectorizer.fit_transform([string])
    tf_features = vectorizer.get_feature_names()
    compose = TruncatedSVD(1).fit(tf)
    return " ".join([tf_features[i] for i in compose.components_[0].argsort()[: -n - 1 : -1]])


# In[4]:


get_ipython().run_cell_magic(
    "time",
    "",
    "h, c = [], []\nfor i in range(len(ctexts)):\n    try:\n        c.append(topic_modelling(ctexts[i]))\n        h.append(headlines[i])\n    except:\n        pass",
)


# In[5]:


len(c), len(h)


# In[6]:


def build_dataset(words, n_words):
    count = [["PAD", 0], ["GO", 1], ["EOS", 2], ["UNK", 3]]
    count.extend(collections.Counter(words).most_common(n_words))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# In[7]:


concat_from = " ".join(c).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(
    concat_from, vocabulary_size_from
)
print("vocab from size: %d" % (vocabulary_size_from))
print("Most common words", count_from[4:10])
print("Sample data", data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])


# In[8]:


concat_to = " ".join(h).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print("vocab to size: %d" % (vocabulary_size_to))
print("Most common words", count_to[4:10])
print("Sample data", data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])


# In[9]:


for i in range(len(h)):
    h[i] = h[i] + " EOS"
h[0]


# In[10]:


GO = dictionary_from["GO"]
PAD = dictionary_from["PAD"]
EOS = dictionary_from["EOS"]
UNK = dictionary_from["UNK"]


# In[11]:


def str_idx(corpus, dic, UNK=3):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k, UNK))
        X.append(ints)
    return X


# In[12]:


X = str_idx(c, dictionary_from)
Y = str_idx(h, dictionary_to)


# In[13]:


train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)


# In[14]:


maxlen_question = max([len(x) for x in X])
maxlen_answer = max([len(y) for y in Y])


# In[15]:


maxlen_question, maxlen_answer


# In[16]:


def layer_normalization(x, epsilon=1e-8):
    shape = x.get_shape()
    tf.Variable(tf.zeros(shape=[int(shape[-1])]))
    beta = tf.Variable(tf.zeros(shape=[int(shape[-1])]))
    gamma = tf.Variable(tf.ones(shape=[int(shape[-1])]))
    mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)
    x = (x - mean) / tf.sqrt(variance + epsilon)
    return gamma * x + beta


def conv1d(input_, output_channels, dilation=1, filter_width=1, causal=False):
    w = tf.Variable(
        tf.random_normal(
            [1, filter_width, int(input_.get_shape()[-1]), output_channels], stddev=0.02
        )
    )
    b = tf.Variable(tf.zeros(shape=[output_channels]))
    if causal:
        padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
        padded = tf.pad(input_, padding)
        input_expanded = tf.expand_dims(padded, dim=1)
        out = tf.nn.atrous_conv2d(input_expanded, w, rate=dilation, padding="VALID") + b
    else:
        input_expanded = tf.expand_dims(input_, dim=1)
        out = tf.nn.atrous_conv2d(input_expanded, w, rate=dilation, padding="SAME") + b
    return tf.squeeze(out, [1])


def bytenet_residual_block(
    input_, dilation, layer_no, residual_channels, filter_width, causal=True
):
    block_type = "decoder" if causal else "encoder"
    block_name = "bytenet_{}_layer_{}_{}".format(block_type, layer_no, dilation)
    with tf.variable_scope(block_name):
        relu1 = tf.nn.relu(layer_normalization(input_))
        conv1 = conv1d(relu1, residual_channels)
        relu2 = tf.nn.relu(layer_normalization(conv1))
        dilated_conv = conv1d(relu2, residual_channels, dilation, filter_width, causal=causal)
        print(dilated_conv)
        relu3 = tf.nn.relu(layer_normalization(dilated_conv))
        conv2 = conv1d(relu3, 2 * residual_channels)
        return input_ + conv2


class ByteNet:
    def __init__(
        self,
        from_vocab_size,
        to_vocab_size,
        channels,
        encoder_dilations,
        decoder_dilations,
        encoder_filter_width,
        decoder_filter_width,
        learning_rate=0.001,
        beta1=0.5,
    ):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        target_1 = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        embedding_channels = 2 * channels
        w_source_embedding = tf.Variable(
            tf.random_normal([from_vocab_size, embedding_channels], stddev=0.02)
        )
        w_target_embedding = tf.Variable(
            tf.random_normal([to_vocab_size, embedding_channels], stddev=0.02)
        )
        source_embedding = tf.nn.embedding_lookup(w_source_embedding, self.X)
        target_1_embedding = tf.nn.embedding_lookup(w_target_embedding, target_1)
        curr_input = source_embedding
        for layer_no, dilation in enumerate(encoder_dilations):
            curr_input = bytenet_residual_block(
                curr_input, dilation, layer_no, channels, encoder_filter_width, causal=False
            )
        encoder_output = curr_input
        combined_embedding = target_1_embedding + encoder_output
        curr_input = combined_embedding
        for layer_no, dilation in enumerate(decoder_dilations):
            curr_input = bytenet_residual_block(
                curr_input, dilation, layer_no, channels, encoder_filter_width, causal=False
            )
        self.logits = conv1d(curr_input, to_vocab_size)
        masks = tf.sequence_mask(self.Y_seq_len, maxlen_question, dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(
            logits=self.logits, targets=self.Y, weights=masks
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[17]:


residual_channels = 128
encoder_dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
decoder_dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
encoder_filter_width = 3
decoder_filter_width = 3
batch_size = 16
epoch = 10


# In[18]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = ByteNet(
    len(dictionary_from),
    len(dictionary_to),
    residual_channels,
    encoder_dilations,
    decoder_dilations,
    encoder_filter_width,
    decoder_filter_width,
)
sess.run(tf.global_variables_initializer())


# In[19]:


def pad_sentence_batch(sentence_batch, pad_int, maxlen):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = maxlen
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(maxlen)
    return padded_seqs, seq_lens


# In[20]:


for EPOCH in range(10):
    lasttime = time.time()
    total_loss, total_accuracy, total_loss_test, total_accuracy_test = 0, 0, 0, 0
    train_X, train_Y = shuffle(train_X, train_Y)
    test_X, test_Y = shuffle(test_X, test_Y)
    pbar = tqdm(range(0, len(train_X), batch_size), desc="train minibatch loop")
    for k in pbar:
        index = min(k + batch_size, len(train_X))
        batch_x, seq_x = pad_sentence_batch(train_X[k:index], PAD, maxlen_question)
        batch_y, seq_y = pad_sentence_batch(train_Y[k:index], PAD, maxlen_question)
        acc, loss, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: batch_y},
        )
        total_loss += loss
        total_accuracy += acc
        pbar.set_postfix(cost=loss, accuracy=acc)

    pbar = tqdm(range(0, len(test_X), batch_size), desc="test minibatch loop")
    for k in pbar:
        batch_x, _ = pad_sentence_batch(
            test_X[k : min(k + batch_size, len(test_X))], PAD, maxlen_question
        )
        batch_y, _ = pad_sentence_batch(
            test_Y[k : min(k + batch_size, len(test_X))], PAD, maxlen_question
        )
        acc, loss = sess.run(
            [model.accuracy, model.cost], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        total_loss_test += loss
        total_accuracy_test += acc
        pbar.set_postfix(cost=loss, accuracy=acc)

    total_loss /= len(train_X) / batch_size
    total_accuracy /= len(train_X) / batch_size
    total_loss_test /= len(test_X) / batch_size
    total_accuracy_test /= len(test_X) / batch_size

    print("epoch: %d, avg loss: %f, avg accuracy: %f" % (EPOCH, total_loss, total_accuracy))
    print(
        "epoch: %d, avg loss test: %f, avg accuracy test: %f"
        % (EPOCH, total_loss_test, total_accuracy_test)
    )


# In[27]:


sess.run(tf.argmax(model.logits, 2), feed_dict={model.X: batch_x, model.Y: batch_y})[0]


# In[ ]:
