#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections

import numpy as np
import tensorflow as tf

# In[2]:


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


def build_dataset(words, n_words):
    count = [["GO", 0], ["PAD", 1], ["EOS", 2], ["UNK", 3]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
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
        target_1 = self.Y[:, :-1]
        target_2 = self.Y[:, 1:]
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
        logits = conv1d(tf.nn.relu(curr_input), to_vocab_size)
        logits_flat = tf.reshape(logits, [-1, to_vocab_size])
        target_flat = tf.reshape(target_2, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_flat, logits=logits_flat
        )
        self.cost = tf.reduce_mean(loss)
        probs_flat = tf.nn.softmax(logits_flat)
        self.t_probs = tf.reshape(probs_flat, [-1, tf.shape(logits)[1], to_vocab_size])
        self.optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.cost)


# In[3]:


with open("fr-en/train.tags.fr-en.fr") as fopen:
    text_from = fopen.read().split("\n")[6:]
with open("fr-en/train.tags.fr-en.en") as fopen:
    text_to = fopen.read().split("\n")[6:]

print("len from: %d, len to: %d" % (len(text_from), len(text_to)))


# In[4]:


concat_from = " ".join(text_from).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(concat_from, 5000)
print("vocab from size: %d" % (vocabulary_size_from))
print("Most common words", count_from[3:10])
print("Sample data", data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])


# In[5]:


concat_to = " ".join(text_to).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, 5000)
print("vocab to size: %d" % (vocabulary_size_to))
print("Most common words", count_to[3:10])
print("Sample data", data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])


# In[6]:


GO = dictionary_from["GO"]
PAD = dictionary_from["PAD"]
EOS = dictionary_from["EOS"]
UNK = dictionary_from["UNK"]


# In[7]:


def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            try:
                ints.append(dic[k])
            except Exception as e:
                ints.append(UNK)
        X.append(ints)
    return X


def create_buckets(text_from, text_to, bucket_quant, from_vocab, to_vocab):
    buckets = {}
    for i in range(len(text_from)):
        text_from[i] = np.concatenate((text_from[i], [from_vocab["EOS"]]))
        text_to[i] = np.concatenate(([to_vocab["GO"]], text_to[i], [to_vocab["EOS"]]))
        sl = len(text_from[i])
        tl = len(text_to[i])
        new_length = max(sl, tl)
        if new_length % bucket_quant > 0:
            new_length = int(((new_length / bucket_quant) + 1) * bucket_quant)

        s_padding = np.array([from_vocab["PAD"] for ctr in range(sl, new_length)])
        t_padding = np.array([to_vocab["PAD"] for ctr in range(tl, new_length + 1)])

        text_from[i] = np.concatenate([text_from[i], s_padding])
        text_to[i] = np.concatenate([text_to[i], t_padding])

        if new_length in buckets:
            buckets[new_length].append((text_from[i], text_to[i]))
        else:
            buckets[new_length] = [(text_from[i], text_to[i])]

    return buckets


# In[8]:


X = str_idx(text_from, dictionary_from)
Y = str_idx(text_to, dictionary_to)


# In[9]:


buckets = create_buckets(X, Y, 50, dictionary_from, dictionary_to)
bucket_sizes = [bucket_size for bucket_size in buckets]
bucket_sizes.sort()


# In[10]:


print("Number Of Buckets: %d" % (len(buckets)))


# In[11]:


residual_channels = 512
encoder_dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
decoder_dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
encoder_filter_width = 3
decoder_filter_width = 3
batch_size = 8
epoch = 1


# In[12]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = ByteNet(
    5005,
    5005,
    residual_channels,
    encoder_dilations,
    decoder_dilations,
    encoder_filter_width,
    decoder_filter_width,
)
sess.run(tf.global_variables_initializer())


# In[ ]:


def get_batch_from_pairs(pair_list):
    source_sentences = []
    target_sentences = []
    for s, t in pair_list:
        source_sentences.append(s)
        target_sentences.append(t)

    return np.array(source_sentences, dtype="int32"), np.array(target_sentences, dtype="int32")


# In[ ]:


for i in range(epoch):
    for bucket_size in bucket_sizes:
        batch_no = 0
        while (batch_no + 1) * batch_size < len(buckets[bucket_size]):
            source, target = get_batch_from_pairs(
                buckets[bucket_size][batch_no * batch_size : (batch_no + 1) * batch_size]
            )
            _, loss = sess.run(
                [model.optimizer, model.cost], feed_dict={model.X: source, model.Y: target}
            )
            if (batch_no + 1) % 50 == 0:
                print(
                    "LOSS %f, batches %d / %d, bucket size %d"
                    % (loss, batch_no + 1, len(buckets[bucket_size]) // batch_size, bucket_size)
                )
            batch_no += 1


# In[ ]:
