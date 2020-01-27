#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

# In[2]:


def build_dataset(words, n_words, atleast=1):
    count = [["PAD", 0], ["GO", 1], ["EOS", 2], ["UNK", 3]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)
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


# In[3]:


lines = open("movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
conv_lines = open("movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

convs = []
for line in conv_lines[:-1]:
    _line = line.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    convs.append(_line.split(","))

questions = []
answers = []

for conv in convs:
    for i in range(len(conv) - 1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i + 1]])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return " ".join([i.strip() for i in filter(None, text.split())])


clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

min_line_length = 2
max_line_length = 5
short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1

question_test = short_questions[500:550]
answer_test = short_answers[500:550]
short_questions = short_questions[:500]
short_answers = short_answers[:500]


# In[4]:


concat_from = " ".join(short_questions + question_test).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(
    concat_from, vocabulary_size_from
)
print("vocab from size: %d" % (vocabulary_size_from))
print("Most common words", count_from[4:10])
print("Sample data", data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])
print("filtered vocab size:", len(dictionary_from))
print("% of vocab used: {}%".format(round(len(dictionary_from) / vocabulary_size_from, 4) * 100))


# In[5]:


concat_to = " ".join(short_answers + answer_test).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print("vocab from size: %d" % (vocabulary_size_to))
print("Most common words", count_to[4:10])
print("Sample data", data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])
print("filtered vocab size:", len(dictionary_to))
print("% of vocab used: {}%".format(round(len(dictionary_to) / vocabulary_size_to, 4) * 100))


# In[6]:


GO = dictionary_from["GO"]
PAD = dictionary_from["PAD"]
EOS = dictionary_from["EOS"]
UNK = dictionary_from["UNK"]


# In[7]:


for i in range(len(short_answers)):
    short_answers[i] += " EOS"


# In[8]:


def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k, UNK))
        X.append(ints)
    return X


def pad_sentence_batch(sentence_batch, pad_int, maxlen):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = maxlen
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(maxlen)
    return padded_seqs, seq_lens


# In[9]:


X = str_idx(short_questions, dictionary_from)
Y = str_idx(short_answers, dictionary_to)
X_test = str_idx(question_test, dictionary_from)
Y_test = str_idx(answer_test, dictionary_from)


# In[10]:


maxlen_question = max([len(x) for x in X]) * 2
maxlen_answer = max([len(y) for y in Y]) * 2


# In[11]:


def hop_forward(memory_o, memory_i, response_proj, inputs_len, questions_len):
    match = memory_i
    match = pre_softmax_masking(match, inputs_len)
    match = tf.nn.softmax(match)
    match = post_softmax_masking(match, questions_len)
    response = tf.multiply(match, memory_o)
    return response_proj(response)


def pre_softmax_masking(x, seq_len):
    paddings = tf.fill(tf.shape(x), float("-inf"))
    T = tf.shape(x)[1]
    max_seq_len = tf.shape(x)[2]
    masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
    masks = tf.tile(tf.expand_dims(masks, 1), [1, T, 1])
    return tf.where(tf.equal(masks, 0), paddings, x)


def post_softmax_masking(x, seq_len):
    T = tf.shape(x)[2]
    max_seq_len = tf.shape(x)[1]
    masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
    masks = tf.tile(tf.expand_dims(masks, -1), [1, 1, T])
    return x * masks


def shift_right(x):
    batch_size = tf.shape(x)[0]
    start = tf.to_int32(tf.fill([batch_size, 1], GO))
    return tf.concat([start, x[:, :-1]], 1)


def embed_seq(x, vocab_size, zero_pad=True):
    lookup_table = tf.get_variable("lookup_table", [vocab_size, size_layer], tf.float32)
    if zero_pad:
        lookup_table = tf.concat((tf.zeros([1, size_layer]), lookup_table[1:, :]), axis=0)
    return tf.nn.embedding_lookup(lookup_table, x)


def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


def quest_mem(x, vocab_size, max_quest_len):
    x = embed_seq(x, vocab_size)
    pos = position_encoding(max_quest_len, size_layer)
    return x * pos


class QA:
    def __init__(self, vocab_size_from, vocab_size_to, size_layer, learning_rate, n_hops=3):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.fill([tf.shape(self.X)[0]], maxlen_question)
        self.Y_seq_len = tf.fill([tf.shape(self.X)[0]], maxlen_answer)
        max_quest_len = maxlen_question
        max_answer_len = maxlen_answer

        lookup_table = tf.get_variable("lookup_table", [vocab_size_from, size_layer], tf.float32)

        with tf.variable_scope("memory_o"):
            memory_o = quest_mem(self.X, vocab_size_from, max_quest_len)

        with tf.variable_scope("memory_i"):
            memory_i = quest_mem(self.X, vocab_size_from, max_quest_len)

        with tf.variable_scope("interaction"):
            response_proj = tf.layers.Dense(size_layer)
            for _ in range(n_hops):
                answer = hop_forward(
                    memory_o, memory_i, response_proj, self.X_seq_len, self.X_seq_len
                )
                memory_i = answer

        embedding = tf.Variable(tf.random_uniform([vocab_size_to, size_layer], -1, 1))
        cell = tf.nn.rnn_cell.BasicRNNCell(size_layer)
        vocab_proj = tf.layers.Dense(vocab_size_to)
        state_proj = tf.layers.Dense(size_layer)
        init_state = state_proj(tf.layers.flatten(answer))

        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tf.nn.embedding_lookup(embedding, shift_right(self.Y)),
            sequence_length=tf.to_int32(self.Y_seq_len),
        )
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell, helper=helper, initial_state=init_state, output_layer=vocab_proj
        )
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, maximum_iterations=max_answer_len
        )

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding,
            start_tokens=tf.tile(tf.constant([GO], dtype=tf.int32), [tf.shape(init_state)[0]]),
            end_token=EOS,
        )
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell, helper=helper, initial_state=init_state, output_layer=vocab_proj
        )
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, maximum_iterations=max_answer_len
        )
        self.training_logits = decoder_output.rnn_output
        self.predicting_ids = predicting_decoder_output.sample_id
        self.logits = decoder_output.sample_id
        masks = tf.sequence_mask(self.Y_seq_len, max_answer_len, dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, targets=self.Y, weights=masks
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[12]:


epoch = 20
batch_size = 16
size_layer = 256

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = QA(len(dictionary_from), len(dictionary_to), size_layer, 1e-3)
sess.run(tf.global_variables_initializer())


# In[13]:


for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    for k in range(0, len(short_questions), batch_size):
        index = min(k + batch_size, len(short_questions))
        batch_x, seq_x = pad_sentence_batch(X[k:index], PAD, maxlen_question)
        batch_y, seq_y = pad_sentence_batch(Y[k:index], PAD, maxlen_answer)
        predicted, accuracy, loss, _ = sess.run(
            [model.predicting_ids, model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: batch_y},
        )
        total_loss += loss
        total_accuracy += accuracy
    total_loss /= len(short_questions) / batch_size
    total_accuracy /= len(short_questions) / batch_size
    print("epoch: %d, avg loss: %f, avg accuracy: %f" % (i + 1, total_loss, total_accuracy))


# In[14]:


for i in range(len(batch_x)):
    print("row %d" % (i + 1))
    print(
        "QUESTION:", " ".join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]])
    )
    print(
        "REAL ANSWER:",
        " ".join([rev_dictionary_to[n] for n in batch_y[i] if n not in [0, 1, 2, 3]]),
    )
    print(
        "PREDICTED ANSWER:",
        " ".join([rev_dictionary_to[n] for n in predicted[i] if n not in [0, 1, 2, 3]]),
        "\n",
    )


# In[15]:


batch_x, seq_x = pad_sentence_batch(X_test[:batch_size], PAD, maxlen_question)
batch_y, seq_y = pad_sentence_batch(Y_test[:batch_size], PAD, maxlen_answer)
predicted = sess.run(model.predicting_ids, feed_dict={model.X: batch_x})

for i in range(len(batch_x)):
    print("row %d" % (i + 1))
    print(
        "QUESTION:", " ".join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]])
    )
    print(
        "REAL ANSWER:",
        " ".join([rev_dictionary_to[n] for n in batch_y[i] if n not in [0, 1, 2, 3]]),
    )
    print(
        "PREDICTED ANSWER:",
        " ".join([rev_dictionary_to[n] for n in predicted[i] if n not in [0, 1, 2, 3]]),
        "\n",
    )


# In[ ]:
