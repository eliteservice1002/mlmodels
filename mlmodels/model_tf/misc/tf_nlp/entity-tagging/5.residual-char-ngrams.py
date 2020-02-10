#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import re
import time

import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

# In[2]:


def process_string(string):
    string = re.sub("[^A-Za-z0-9\-\/ ]+", " ", string).split()
    return [y.strip() for y in string]


def to_title(string):
    if string.isupper():
        string = string.title()
    return string


# In[3]:


def parse_raw(filename):
    with open(filename, "r") as fopen:
        entities = fopen.read()
    soup = BeautifulSoup(entities, "html.parser")
    inside_tag = ""
    texts, labels = [], []
    for sentence in soup.prettify().split("\n"):
        if len(inside_tag):
            splitted = process_string(sentence)
            texts += splitted
            labels += [inside_tag] * len(splitted)
            inside_tag = ""
        else:
            if not sentence.find("</"):
                pass
            elif not sentence.find("<"):
                inside_tag = sentence.split(">")[0][1:]
            else:
                splitted = process_string(sentence)
                texts += splitted
                labels += ["OTHER"] * len(splitted)
    assert len(texts) == len(labels), "length texts and labels are not same"
    print("len texts and labels: ", len(texts))
    return texts, labels


# In[4]:


train_texts, train_labels = parse_raw("data_train.txt")
test_texts, test_labels = parse_raw("data_test.txt")
train_texts += test_texts
train_labels += test_labels


# In[5]:


with open("entities-bm-normalize-v3.txt", "r") as fopen:
    entities_bm = fopen.read().split("\n")[:-1]
entities_bm = [i.split() for i in entities_bm]
entities_bm = [[i[0], "TIME" if i[0] in "jam" else i[1]] for i in entities_bm]


# In[6]:


replace_by = {"organizaiton": "organization", "orgnization": "organization", "othoer": "OTHER"}

with open("NER-part1.txt", "r") as fopen:
    nexts = fopen.read().split("\n")[:-1]
nexts = [i.split() for i in nexts]
for i in nexts:
    if len(i) == 2:
        label = i[1].lower()
        if "other" in label:
            label = label.upper()
        if label in replace_by:
            label = replace_by[label]
        train_labels.append(label)
        train_texts.append(i[0])


# In[7]:


replace_by = {
    "LOC": "location",
    "PRN": "person",
    "NORP": "organization",
    "ORG": "organization",
    "LAW": "law",
    "EVENT": "event",
    "FAC": "organization",
    "TIME": "time",
    "O": "OTHER",
    "ART": "person",
    "DOC": "law",
}
for i in entities_bm:
    try:
        string = process_string(i[0])
        if len(string):
            train_labels.append(replace_by[i[1]])
            train_texts.append(process_string(i[0])[0])
    except Exception as e:
        print(e)

assert len(train_texts) == len(train_labels), "length texts and labels are not same"


# In[8]:


np.unique(train_labels, return_counts=True)


# In[9]:


def _pad_sequence(
    sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None
):
    sequence = iter(sequence)
    if pad_left:
        sequence = itertools.chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = itertools.chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(
    sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None
):
    """
    generate ngrams

    Parameters
    ----------
    sequence : list of str
        list of tokenize words
    n : int
        ngram size

    Returns
    -------
    ngram: list
    """
    sequence = _pad_sequence(sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


# In[10]:


def get_ngrams(s, grams=(2, 3, 4)):
    return ["".join(i) for k in grams for i in list(ngrams(s, k))]


# In[11]:


word2idx = {"PAD": 0, "NUM": 1, "UNK": 2}
tag2idx = {"PAD": 0}
char2idx = {"PAD": 0, "NUM": 1, "UNK": 2}
word_idx = 3
tag_idx = 1
char_idx = 3


def parse_XY(texts, labels):
    global word2idx, tag2idx, char2idx, word_idx, tag_idx, char_idx
    X, Y = [], []
    for no, text in enumerate(texts):
        text = text.lower()
        if len(text) < 2:
            continue
        tag = labels[no]
        for c in get_ngrams(text):
            if c not in char2idx:
                char2idx[c] = char_idx
                char_idx += 1
        if tag not in tag2idx:
            tag2idx[tag] = tag_idx
            tag_idx += 1
        Y.append(tag2idx[tag])
        if text not in word2idx:
            word2idx[text] = word_idx
            word_idx += 1
        X.append(word2idx[text])
    return X, np.array(Y)


# In[12]:


X, Y = parse_XY(train_texts, train_labels)
idx2word = {idx: tag for tag, idx in word2idx.items()}
idx2tag = {i: w for w, i in tag2idx.items()}


# In[13]:


seq_len = 50


def iter_seq(x):
    return np.array([x[i : i + seq_len] for i in range(0, len(x) - seq_len, 1)])


def to_train_seq(*args):
    return [iter_seq(x) for x in args]


def generate_char_seq(batch, maxlen):
    temp = np.zeros((batch.shape[0], batch.shape[1], maxlen), dtype=np.int32)
    for i in range(batch.shape[0]):
        for k in range(batch.shape[1]):
            for no, c in enumerate(get_ngrams(idx2word[batch[i, k]])):
                temp[i, k, no] = char2idx[c]
    return temp


# In[14]:


X_seq, Y_seq = to_train_seq(X, Y)
X_char_seq = generate_char_seq(X_seq, seq_len * 2)
X_seq.shape


# In[15]:


Y_seq.shape


# In[16]:


train_Y, test_Y, train_X, test_X = train_test_split(Y_seq, X_char_seq, test_size=0.2)


# In[17]:


class Attention:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.dense_layer = tf.layers.Dense(hidden_size)
        self.v = tf.random_normal([hidden_size], mean=0, stddev=1 / np.sqrt(hidden_size))

    def score(self, hidden_tensor, encoder_outputs):
        energy = tf.nn.tanh(self.dense_layer(tf.concat([hidden_tensor, encoder_outputs], 2)))
        energy = tf.transpose(energy, [0, 2, 1])
        batch_size = tf.shape(encoder_outputs)[0]
        v = tf.expand_dims(tf.tile(tf.expand_dims(self.v, 0), [batch_size, 1]), 1)
        energy = tf.matmul(v, energy)
        return tf.squeeze(energy, 1)

    def __call__(self, hidden, encoder_outputs):
        seq_len = tf.shape(encoder_outputs)[1]
        batch_size = tf.shape(encoder_outputs)[0]
        H = tf.tile(tf.expand_dims(hidden, 1), [1, seq_len, 1])
        attn_energies = self.score(H, encoder_outputs)
        return tf.expand_dims(tf.nn.softmax(attn_energies), 1)


class Model:
    def __init__(self, dict_size, size_layers, learning_rate, maxlen, num_blocks=3, block_size=128):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, maxlen, maxlen * 2])
        self.labels = tf.placeholder(tf.int32, shape=[None, maxlen])
        embeddings = tf.Variable(tf.random_uniform([dict_size, size_layers], -1, 1))
        embedded = tf.nn.embedding_lookup(embeddings, self.word_ids)
        embedded = tf.reduce_mean(embedded, axis=2)
        self.attention = Attention(size_layers)
        self.maxlen = tf.shape(self.word_ids)[1]
        self.lengths = tf.count_nonzero(tf.reduce_sum(self.word_ids, axis=2), 1)

        def residual_block(x, size, rate, block):
            with tf.variable_scope("block_%d_%d" % (block, rate), reuse=False):
                attn_weights = self.attention(tf.reduce_sum(x, axis=1), x)
                conv_filter = tf.layers.conv1d(
                    attn_weights,
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

        forward = tf.layers.conv1d(embedded, block_size, kernel_size=1, strides=1, padding="SAME")
        zeros = tf.zeros_like(forward)
        for i in range(num_blocks):
            for r in [1, 2, 4, 8, 16]:
                forward, s = residual_block(forward, size=7, rate=r, block=i)
                zeros = tf.add(zeros, s)
        logits = tf.layers.conv1d(zeros, len(idx2tag), kernel_size=1, strides=1, padding="SAME")
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, self.labels, self.lengths
        )
        self.cost = tf.reduce_mean(-log_likelihood)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        mask = tf.sequence_mask(self.lengths, maxlen=self.maxlen)

        self.tags_seq, _ = tf.contrib.crf.crf_decode(logits, transition_params, self.lengths)

        self.prediction = tf.boolean_mask(self.tags_seq, mask)
        mask_label = tf.boolean_mask(self.labels, mask)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[18]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

dim = 256
dropout = 1
learning_rate = 1e-3
batch_size = 32

model = Model(len(char2idx), dim, learning_rate, seq_len)
sess.run(tf.global_variables_initializer())


# In[19]:


for e in range(2):
    lasttime = time.time()
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    pbar = tqdm(range(0, len(train_X), batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x = train_X[i : min(i + batch_size, train_X.shape[0])]
        batch_y = train_Y[i : min(i + batch_size, train_X.shape[0])]
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.word_ids: batch_x, model.labels: batch_y},
        )
        assert not np.isnan(cost)
        train_loss += cost
        train_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    pbar = tqdm(range(0, len(test_X), batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        acc, cost = sess.run(
            [model.accuracy, model.cost], feed_dict={model.word_ids: batch_x, model.labels: batch_y}
        )
        assert not np.isnan(cost)
        test_loss += cost
        test_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    train_loss /= len(train_X) / batch_size
    train_acc /= len(train_X) / batch_size
    test_loss /= len(test_X) / batch_size
    test_acc /= len(test_X) / batch_size

    print("time taken:", time.time() - lasttime)
    print(
        "epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n"
        % (e, train_loss, train_acc, test_loss, test_acc)
    )


# In[ ]:
