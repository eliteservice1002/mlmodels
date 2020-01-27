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


def generate_char_seq(batch):
    x = [[len(get_ngrams(idx2word[i])) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((batch.shape[0], batch.shape[1], maxlen), dtype=np.int32)
    for i in range(batch.shape[0]):
        for k in range(batch.shape[1]):
            for no, c in enumerate(get_ngrams(idx2word[batch[i, k]])):
                temp[i, k, no] = char2idx[c]
    return temp


# In[14]:


X_seq, Y_seq = to_train_seq(X, Y)
X_char_seq = generate_char_seq(X_seq)
X_seq.shape


# In[15]:


X_char_seq.shape


# In[16]:


train_Y, test_Y, train_X, test_X = train_test_split(Y_seq, X_char_seq, test_size=0.2)


# In[17]:


class Model:
    def __init__(self, dim_word, dropout, learning_rate, hidden_size_word, num_layers):
        def cells(size, reuse=False):
            return tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(), reuse=reuse),
                output_keep_prob=dropout,
            )

        def bahdanau(embedded, size):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=hidden_size_word, memory=embedded
            )
            return tf.contrib.seq2seq.AttentionWrapper(
                cell=cells(hidden_size_word),
                attention_mechanism=attention_mechanism,
                attention_layer_size=hidden_size_word,
            )

        self.word_ids = tf.placeholder(tf.int32, shape=[None, None, None])
        self.labels = tf.placeholder(tf.int32, shape=[None, None])
        self.maxlen = tf.shape(self.word_ids)[1]
        self.lengths = tf.count_nonzero(tf.reduce_sum(self.word_ids, axis=2), 1)

        self.word_embeddings = tf.Variable(
            tf.truncated_normal([len(char2idx), dim_word], stddev=1.0 / np.sqrt(dim_word))
        )
        word_embedded = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)
        word_embedded = tf.reduce_mean(word_embedded, axis=2)

        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=bahdanau(word_embedded, hidden_size_word),
                cell_bw=bahdanau(word_embedded, hidden_size_word),
                inputs=word_embedded,
                dtype=tf.float32,
                scope="bidirectional_rnn_word_%d" % (n),
            )
            word_embedded = tf.concat((out_fw, out_bw), 2)

        logits = tf.layers.dense(word_embedded, len(idx2tag))
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, self.labels, self.lengths
        )
        self.cost = tf.reduce_mean(-log_likelihood)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        mask = tf.sequence_mask(self.lengths, maxlen=self.maxlen)
        self.tags_seq, tags_score = tf.contrib.crf.crf_decode(
            logits, transition_params, self.lengths
        )
        self.tags_seq = tf.identity(self.tags_seq, name="logits")

        self.prediction = tf.boolean_mask(self.tags_seq, mask)
        mask_label = tf.boolean_mask(self.labels, mask)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[18]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

dim_word = 128
dropout = 0.8
learning_rate = 1e-3
hidden_size_word = 64
num_layers = 2
batch_size = 32

model = Model(dim_word, dropout, learning_rate, hidden_size_word, num_layers)
sess.run(tf.global_variables_initializer())


# In[19]:


for e in range(3):
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


# In[20]:


string = "KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu."


# In[21]:


def generate_char_seq(batch, UNK=2):
    maxlen_c = max([len(k) for k in batch])
    x = [[len(get_ngrams(i)) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((len(batch), maxlen_c, maxlen), dtype=np.int32)
    for i in range(len(batch)):
        for k in range(len(batch[i])):
            for no, c in enumerate(get_ngrams(batch[i][k])):
                temp[i, k, no] = char2idx.get(c, UNK)
    return temp


sequence = process_string(string.lower())
X_char_seq = generate_char_seq([sequence])


# In[22]:


X_char_seq.shape


# In[23]:


predicted = sess.run(model.tags_seq, feed_dict={model.word_ids: X_char_seq})[0]
for i in range(len(predicted)):
    print(sequence[i], idx2tag[predicted[i]])


# In[ ]:
