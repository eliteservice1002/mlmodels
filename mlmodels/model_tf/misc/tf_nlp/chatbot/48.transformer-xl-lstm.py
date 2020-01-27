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


learning_rate = 1e-3
batch_size = 16
epoch = 20
n_layer = 3
d_model = 256
d_embed = 256
n_head = 10
d_head = 50
d_inner = 512


# In[12]:


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum("i,j->ij", pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :]


def positionwise_FF(inp, d_model, d_inner, kernel_initializer, scope="ff"):
    output = inp
    with tf.variable_scope(scope):
        output = tf.layers.dense(
            inp,
            d_inner,
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            name="layer_1",
        )
        output = tf.layers.dense(
            output, d_model, kernel_initializer=kernel_initializer, name="layer_2"
        )
        output = tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1)
    return output


def rel_shift(x):
    x_size = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)

    return x


def rel_multihead_attn(
    w,
    r,
    r_w_bias,
    r_r_bias,
    attn_mask,
    mems,
    d_model,
    n_head,
    d_head,
    kernel_initializer,
    scope="rel_attn",
):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        rlen = tf.shape(r)[0]
        bsz = tf.shape(w)[1]

        cat = tf.concat([mems, w], 0) if mems is not None and mems.shape.ndims > 1 else w
        w_heads = tf.layers.dense(
            cat,
            3 * n_head * d_head,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name="qkv",
        )
        r_head_k = tf.layers.dense(
            r, n_head * d_head, use_bias=False, kernel_initializer=kernel_initializer, name="r"
        )

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

        r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

        rw_head_q = w_head_q + r_w_bias
        rr_head_q = w_head_q + r_r_bias

        AC = tf.einsum("ibnd,jbnd->ijbn", rw_head_q, w_head_k)
        BD = tf.einsum("ibnd,jnd->ijbn", rr_head_q, r_head_k)
        BD = rel_shift(BD)

        paddings = tf.fill(tf.shape(BD), float("-inf"))

        attn_score = (AC + BD) * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(
            attn_vec, d_model, use_bias=False, kernel_initializer=kernel_initializer, name="o"
        )

        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output


def embedding_lookup(lookup_table, x):
    return tf.nn.embedding_lookup(lookup_table, x)


def mask_adaptive_embedding_lookup(
    x,
    n_token,
    d_embed,
    d_proj,
    cutoffs,
    initializer,
    proj_initializer,
    div_val=1,
    proj_same_dim=True,
    scope="adaptive_embed",
    **kwargs,
):
    emb_scale = d_proj ** 0.5
    with tf.variable_scope(scope):
        if div_val == 1:
            lookup_table = tf.get_variable(
                "lookup_table", [n_token, d_embed], initializer=initializer
            )
            y = embedding_lookup(lookup_table, x)
            if d_proj != d_embed:
                proj_W = tf.get_variable("proj_W", [d_embed, d_proj], initializer=proj_initializer)
                y = tf.einsum("ibe,ed->ibd", y, proj_W)
            else:
                proj_W = None
            ret_params = [lookup_table, proj_W]
        else:
            tables, projs = [], []
            cutoff_ends = [0] + cutoffs + [n_token]
            x_size = tf.shape(x)
            y = tf.zeros([x_size[0], x_size[1], d_proj])
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope("cutoff_{}".format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    mask = (x >= l_idx) & (x < r_idx)
                    cur_x = tf.boolean_mask(x, mask) - l_idx
                    cur_d_embed = d_embed // (div_val ** i)
                    lookup_table = tf.get_variable(
                        "lookup_table", [r_idx - l_idx, cur_d_embed], initializer=initializer
                    )
                    cur_y = embedding_lookup(lookup_table, cur_x)
                    if d_proj == cur_d_embed and not proj_same_dim:
                        proj_W = None
                    else:
                        proj_W = tf.get_variable(
                            "proj_W", [cur_d_embed, d_proj], initializer=proj_initializer
                        )
                        cur_y = tf.einsum("id,de->ie", cur_y, proj_W)
                    mask_idx = tf.to_int64(tf.where(mask))
                    y += tf.scatter_nd(mask_idx, cur_y, tf.to_int64(tf.shape(y)))
                    tables.append(lookup_table)
                    projs.append(proj_W)
            ret_params = [tables, projs]

    y *= emb_scale
    return y, ret_params


def _create_mask(qlen, mlen, same_length=False):
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.matrix_band_part(attn_mask, 0, -1)
    mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


def _cache_mem(curr_out, prev_mem, mem_len=None):
    if mem_len is None or prev_mem is None:
        new_mem = curr_out
    elif mem_len == 0:
        return prev_mem
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]

    return tf.stop_gradient(new_mem)


def transformer(
    dec_inp,
    mems,
    n_token,
    n_layer,
    d_model,
    d_embed,
    n_head,
    d_head,
    d_inner,
    initializer,
    proj_initializer=None,
    mem_len=None,
    cutoffs=[],
    div_val=1,
    tie_projs=[],
    same_length=False,
    clamp_len=-1,
    untie_r=False,
    proj_same_dim=True,
    scope="transformer",
    reuse=tf.AUTO_REUSE,
):
    """
  cutoffs: a list of python int. Cutoffs for adaptive softmax.
  tie_projs: a list of python bools. Whether to tie the projections.
  perms: a list of tensors. Each tensor should of size [len, bsz, bin_size].
        Only used in the adaptive setting.
  """
    new_mems = []
    with tf.variable_scope(scope, reuse=reuse):
        if untie_r:
            r_w_bias = tf.get_variable(
                "r_w_bias", [n_layer, n_head, d_head], initializer=initializer
            )
            r_r_bias = tf.get_variable(
                "r_r_bias", [n_layer, n_head, d_head], initializer=initializer
            )
        else:
            r_w_bias = tf.get_variable("r_w_bias", [n_head, d_head], initializer=initializer)
            r_r_bias = tf.get_variable("r_r_bias", [n_head, d_head], initializer=initializer)

        qlen = tf.shape(dec_inp)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen

        if proj_initializer is None:
            proj_initializer = initializer
        lookup_fn = mask_adaptive_embedding_lookup
        embeddings, shared_params = lookup_fn(
            x=dec_inp,
            n_token=n_token,
            d_embed=d_embed,
            d_proj=d_model,
            cutoffs=cutoffs,
            initializer=initializer,
            proj_initializer=proj_initializer,
            div_val=div_val,
            proj_same_dim=proj_same_dim,
        )

        attn_mask = _create_mask(qlen, mlen, same_length)

        pos_seq = tf.range(klen - 1, -1, -1.0)
        if clamp_len > 0:
            pos_seq = tf.minimum(pos_seq, clamp_len)
        inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
        pos_emb = positional_embedding(pos_seq, inv_freq)

        if mems is None:
            mems = [None] * n_layer
        output = embeddings
        for i in range(n_layer):
            # cache new mems
            new_mems.append(_cache_mem(output, mems[i], mem_len))

            with tf.variable_scope("layer_{}".format(i)):
                output = rel_multihead_attn(
                    w=output,
                    r=pos_emb,
                    r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                    r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                    attn_mask=attn_mask,
                    mems=mems[i],
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    kernel_initializer=initializer,
                )
                output = positionwise_FF(
                    inp=output, d_model=d_model, d_inner=d_inner, kernel_initializer=initializer
                )

        return output, new_mems


# In[13]:


class Chatbot:
    def __init__(self):

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])

        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        initializer = tf.initializers.random_normal(stddev=0.1)
        memory = tf.fill([n_layer, tf.shape(self.X)[0], tf.shape(self.X)[1], d_model], PAD)
        memory = tf.cast(memory, tf.float32)
        logits, next_memory = transformer(
            self.X,
            memory,
            len(dictionary_from),
            n_layer,
            d_model,
            d_embed,
            n_head,
            d_head,
            d_inner,
            initializer,
            scope="encoder",
            reuse=False,
        )
        logits, next_memory = transformer(
            self.X,
            next_memory,
            len(dictionary_from),
            n_layer,
            d_model,
            d_embed,
            n_head,
            d_head,
            d_inner,
            initializer,
            scope="decoder",
            reuse=False,
        )
        embedding = tf.Variable(tf.random_uniform([len(dictionary_to), d_model], -1, 1))
        init_state = tf.reduce_mean(tf.reduce_mean(next_memory, axis=0), axis=1)
        cell = tf.nn.rnn_cell.LSTMCell(d_model)
        vocab_proj = tf.layers.Dense(len(dictionary_to))
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tf.nn.embedding_lookup(embedding, decoder_input),
            sequence_length=tf.to_int32(self.Y_seq_len),
        )
        encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=init_state, h=init_state)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell, helper=helper, initial_state=encoder_state, output_layer=vocab_proj
        )
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, maximum_iterations=tf.reduce_max(self.Y_seq_len)
        )

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding,
            start_tokens=tf.tile(tf.constant([GO], dtype=tf.int32), [tf.shape(init_state)[0]]),
            end_token=EOS,
        )
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell, helper=helper, initial_state=encoder_state, output_layer=vocab_proj
        )
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, maximum_iterations=2 * tf.reduce_max(self.X_seq_len)
        )
        self.training_logits = decoder_output.rnn_output
        self.predicting_ids = predicting_decoder_output.sample_id
        self.logits = decoder_output.sample_id
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
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


# In[14]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot()
sess.run(tf.global_variables_initializer())


# In[15]:


def pad_sentence_batch_dynamic(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


# In[16]:


for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    for k in range(0, len(short_questions), batch_size):
        index = min(k + batch_size, len(short_questions))
        batch_x, seq_x = pad_sentence_batch_dynamic(X[k:index], PAD)
        batch_y, seq_y = pad_sentence_batch_dynamic(Y[k:index], PAD)
        predicted, accuracy, loss, _ = sess.run(
            [model.predicting_ids, model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: batch_y},
        )
        total_loss += loss
        total_accuracy += accuracy
    total_loss /= len(short_questions) / batch_size
    total_accuracy /= len(short_questions) / batch_size
    print("epoch: %d, avg loss: %f, avg accuracy: %f" % (i + 1, total_loss, total_accuracy))


# In[17]:


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


# In[18]:


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
