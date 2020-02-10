#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm

from keras.utils import to_categorical

# In[2]:


def process_string(string):
    string = re.sub("[^A-Za-z0-9\-\/ ]+", " ", string).split()
    return [to_title(y.strip()) for y in string]


def to_title(string):
    if string.isupper():
        string = string.title()
    return string


# In[3]:


with open("pos-data-v3.json", "r") as fopen:
    dataset = json.load(fopen)


# In[4]:


texts, labels = [], []
for i in dataset:
    try:
        texts.append(process_string(i[0])[0].lower())
        labels.append(i[-1])
    except Exception as e:
        print(e, i)


# In[5]:


word2idx = {"PAD": 0, "NUM": 1, "UNK": 2}
tag2idx = {"PAD": 0}
char2idx = {"PAD": 0}
word_idx = 3
tag_idx = 1
char_idx = 1


def parse_XY(texts, labels):
    global word2idx, tag2idx, char2idx, word_idx, tag_idx, char_idx
    X, Y = [], []
    for no, text in enumerate(texts):
        text = to_title(text)
        tag = labels[no]
        for c in text:
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


# In[6]:


X, Y = parse_XY(texts, labels)
idx2word = {idx: tag for tag, idx in word2idx.items()}
idx2tag = {i: w for w, i in tag2idx.items()}


# In[7]:


seq_len = 50


def iter_seq(x):
    return np.array([x[i : i + seq_len] for i in range(0, len(x) - seq_len, 1)])


def to_train_seq(*args):
    return [iter_seq(x) for x in args]


def generate_char_seq(batch):
    x = [[len(idx2word[i]) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((batch.shape[0], batch.shape[1], maxlen), dtype=np.int32)
    for i in range(batch.shape[0]):
        for k in range(batch.shape[1]):
            for no, c in enumerate(idx2word[batch[i, k]]):
                temp[i, k, -1 - no] = char2idx[c]
    return temp


# In[8]:


X_seq, Y_seq = to_train_seq(X, Y)
X_char_seq = generate_char_seq(X_seq)
X_seq.shape


# In[9]:


with open("luong-pos.json", "w") as fopen:
    fopen.write(
        json.dumps(
            {
                "idx2tag": idx2tag,
                "idx2word": idx2word,
                "word2idx": word2idx,
                "tag2idx": tag2idx,
                "char2idx": char2idx,
            }
        )
    )


# In[10]:


Y_seq_3d = [to_categorical(i, num_classes=len(tag2idx)) for i in Y_seq]


# In[11]:


train_X, test_X, train_Y, test_Y, train_char, test_char = train_test_split(
    X_seq, Y_seq_3d, X_char_seq, test_size=0.1
)


# In[12]:


class Model:
    def __init__(
        self,
        dim_word,
        dim_char,
        dropout,
        learning_rate,
        hidden_size_char,
        hidden_size_word,
        num_layers,
    ):
        def cells(size, reuse=False):
            return tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(), reuse=reuse),
                state_keep_prob=dropout,
                output_keep_prob=dropout,
            )

        def luong(embedded, size):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=hidden_size_word, memory=embedded
            )
            return tf.contrib.seq2seq.AttentionWrapper(
                cell=cells(hidden_size_word),
                attention_mechanism=attention_mechanism,
                attention_layer_size=hidden_size_word,
            )

        self.word_ids = tf.placeholder(tf.int32, shape=[None, None])
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None])
        self.labels = tf.placeholder(tf.int32, shape=[None, None, None])
        self.maxlen = tf.shape(self.word_ids)[1]
        self.lengths = tf.count_nonzero(self.word_ids, 1)

        self.word_embeddings = tf.Variable(
            tf.truncated_normal([len(word2idx), dim_word], stddev=1.0 / np.sqrt(dim_word))
        )
        self.char_embeddings = tf.Variable(
            tf.truncated_normal([len(char2idx), dim_char], stddev=1.0 / np.sqrt(dim_char))
        )

        word_embedded = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)
        char_embedded = tf.nn.embedding_lookup(self.char_embeddings, self.char_ids)
        s = tf.shape(char_embedded)
        char_embedded = tf.reshape(char_embedded, shape=[s[0] * s[1], s[-2], dim_char])

        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells(hidden_size_char),
                cell_bw=cells(hidden_size_char),
                inputs=char_embedded,
                dtype=tf.float32,
                scope="bidirectional_rnn_char_%d" % (n),
            )
            char_embedded = tf.concat((out_fw, out_bw), 2)
        output = tf.reshape(char_embedded[:, -1], shape=[s[0], s[1], 2 * hidden_size_char])
        word_embedded = tf.concat([word_embedded, output], axis=-1)

        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=luong(word_embedded, hidden_size_word),
                cell_bw=luong(word_embedded, hidden_size_word),
                inputs=word_embedded,
                dtype=tf.float32,
                scope="bidirectional_rnn_word_%d" % (n),
            )
            word_embedded = tf.concat((out_fw, out_bw), 2)

        logits = tf.layers.dense(word_embedded, len(idx2tag))
        y_t = tf.argmax(self.labels, 2)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, y_t, self.lengths
        )
        self.cost = tf.reduce_mean(-log_likelihood)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        mask = tf.sequence_mask(self.lengths, maxlen=self.maxlen)
        self.tags_seq, tags_score = tf.contrib.crf.crf_decode(
            logits, transition_params, self.lengths
        )
        self.tags_seq = tf.identity(self.tags_seq, name="logits")

        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(self.tags_seq, mask)
        mask_label = tf.boolean_mask(y_t, mask)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[13]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

dim_word = 128
dim_char = 256
dropout = 0.8
learning_rate = 1e-3
hidden_size_char = 64
hidden_size_word = 64
num_layers = 2
batch_size = 32

model = Model(
    dim_word, dim_char, dropout, learning_rate, hidden_size_char, hidden_size_word, num_layers
)
sess.run(tf.global_variables_initializer())


# In[14]:


for e in range(2):
    lasttime = time.time()
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    pbar = tqdm(range(0, len(train_X), batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x = train_X[i : min(i + batch_size, train_X.shape[0])]
        batch_char = train_char[i : min(i + batch_size, train_X.shape[0])]
        batch_y = train_Y[i : min(i + batch_size, train_X.shape[0])]
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.word_ids: batch_x, model.char_ids: batch_char, model.labels: batch_y},
        )
        assert not np.isnan(cost)
        train_loss += cost
        train_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    pbar = tqdm(range(0, len(test_X), batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
        batch_char = test_char[i : min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        acc, cost = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.word_ids: batch_x, model.char_ids: batch_char, model.labels: batch_y},
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


# In[15]:


def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p])
        out.append(out_i)
    return out


# In[16]:


real_Y, predict_Y = [], []

pbar = tqdm(range(0, len(test_X), batch_size), desc="validation minibatch loop")
for i in pbar:
    batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
    batch_char = test_char[i : min(i + batch_size, test_X.shape[0])]
    batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
    predicted = pred2label(
        sess.run(model.tags_seq, feed_dict={model.word_ids: batch_x, model.char_ids: batch_char})
    )
    real = pred2label(np.argmax(batch_y, axis=2))
    predict_Y.extend(predicted)
    real_Y.extend(real)


# In[17]:


print(classification_report(np.array(real_Y).ravel(), np.array(predict_Y).ravel()))


# In[18]:


saver = tf.train.Saver(tf.trainable_variables())
saver.save(sess, "concat-bidirectional-luong-pos/model.ckpt")

strings = ",".join(
    [
        n.name
        for n in tf.get_default_graph().as_graph_def().node
        if (
            "Variable" in n.op
            or "Placeholder" in n.name
            or "logits" in n.name
            or "alphas" in n.name
        )
        and "Adam" not in n.name
        and "beta" not in n.name
        and "OptimizeLoss" not in n.name
        and "Global_Step" not in n.name
    ]
)
strings.split(",")


# In[19]:


def freeze_graph(model_dir, output_node_names):

    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export " "directory: %s" % model_dir
        )

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    absolute_model_dir = "/".join(input_checkpoint.split("/")[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"
    clear_devices = True
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + ".meta", clear_devices=clear_devices)
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), output_node_names.split(",")
        )
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


# In[20]:


freeze_graph("concat-bidirectional-luong-pos", strings)


# In[21]:


g = load_graph("concat-bidirectional-luong-pos/frozen_model.pb")


# In[22]:


string = "KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu."


# In[23]:


def char_str_idx(corpus, dic, UNK=0):
    maxlen = max([len(i) for i in corpus])
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i][:maxlen][::-1]):
            val = dic[k] if k in dic else UNK
            X[i, -1 - no] = val
    return X


def generate_char_seq(batch, idx2word, char2idx):
    x = [[len(idx2word[i]) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((batch.shape[0], batch.shape[1], maxlen), dtype=np.int32)
    for i in range(batch.shape[0]):
        for k in range(batch.shape[1]):
            for no, c in enumerate(idx2word[batch[i, k]].lower()):
                temp[i, k, -1 - no] = char2idx[c]
    return temp


sequence = process_string(string)
X_seq = char_str_idx([sequence], word2idx, 2)
X_char_seq = generate_char_seq(X_seq, idx2word, char2idx)


# In[24]:


word_ids = g.get_tensor_by_name("import/Placeholder:0")
char_ids = g.get_tensor_by_name("import/Placeholder_1:0")
tags_seq = g.get_tensor_by_name("import/logits:0")
test_sess = tf.InteractiveSession(graph=g)
predicted = test_sess.run(tags_seq, feed_dict={word_ids: X_seq, char_ids: X_char_seq})[0]

for i in range(len(predicted)):
    print(sequence[i], idx2tag[predicted[i]])


# In[ ]:
