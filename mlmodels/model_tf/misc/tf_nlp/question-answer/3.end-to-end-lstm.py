#!/usr/bin/env python
# coding: utf-8

# In[1]:


from copy import deepcopy

import numpy as np
import tensorflow as tf

# In[2]:


epoch = 20
batch_size = 64
size_layer = 64
dropout_rate = 0.5
n_hops = 2


# In[3]:


class BaseDataLoader:
    def __init__(self):
        self.data = {
            "size": None,
            "val": {"inputs": None, "questions": None, "answers": None},
            "len": {
                "inputs_len": None,
                "inputs_sent_len": None,
                "questions_len": None,
                "answers_len": None,
            },
        }
        self.vocab = {"size": None, "word2idx": None, "idx2word": None}
        self.params = {
            "vocab_size": None,
            "<start>": None,
            "<end>": None,
            "max_input_len": None,
            "max_sent_len": None,
            "max_quest_len": None,
            "max_answer_len": None,
        }


class DataLoader(BaseDataLoader):
    def __init__(self, path, is_training, vocab=None, params=None):
        super().__init__()
        data, lens = self.load_data(path)
        if is_training:
            self.build_vocab(data)
        else:
            self.demo = data
            self.vocab = vocab
            self.params = deepcopy(params)
        self.is_training = is_training
        self.padding(data, lens)

    def load_data(self, path):
        data, lens = bAbI_data_load(path)
        self.data["size"] = len(data[0])
        return data, lens

    def build_vocab(self, data):
        signals = ["<pad>", "<unk>", "<start>", "<end>"]
        inputs, questions, answers = data
        i_words = [w for facts in inputs for fact in facts for w in fact if w != "<end>"]
        q_words = [w for question in questions for w in question]
        a_words = [w for answer in answers for w in answer if w != "<end>"]
        words = list(set(i_words + q_words + a_words))
        self.params["vocab_size"] = len(words) + 4
        self.params["<start>"] = 2
        self.params["<end>"] = 3
        self.vocab["word2idx"] = {word: idx for idx, word in enumerate(signals + words)}
        self.vocab["idx2word"] = {idx: word for word, idx in self.vocab["word2idx"].items()}

    def padding(self, data, lens):
        inputs_len, inputs_sent_len, questions_len, answers_len = lens

        self.params["max_input_len"] = max(inputs_len)
        self.params["max_sent_len"] = max(
            [fact_len for batch in inputs_sent_len for fact_len in batch]
        )
        self.params["max_quest_len"] = max(questions_len)
        self.params["max_answer_len"] = max(answers_len)

        self.data["len"]["inputs_len"] = np.array(inputs_len)
        for batch in inputs_sent_len:
            batch += [0] * (self.params["max_input_len"] - len(batch))
        self.data["len"]["inputs_sent_len"] = np.array(inputs_sent_len)
        self.data["len"]["questions_len"] = np.array(questions_len)
        self.data["len"]["answers_len"] = np.array(answers_len)

        inputs, questions, answers = deepcopy(data)
        for facts in inputs:
            for sentence in facts:
                for i in range(len(sentence)):
                    sentence[i] = self.vocab["word2idx"].get(
                        sentence[i], self.vocab["word2idx"]["<unk>"]
                    )
                sentence += [0] * (self.params["max_sent_len"] - len(sentence))
            paddings = [0] * self.params["max_sent_len"]
            facts += [paddings] * (self.params["max_input_len"] - len(facts))
        for question in questions:
            for i in range(len(question)):
                question[i] = self.vocab["word2idx"].get(
                    question[i], self.vocab["word2idx"]["<unk>"]
                )
            question += [0] * (self.params["max_quest_len"] - len(question))
        for answer in answers:
            for i in range(len(answer)):
                answer[i] = self.vocab["word2idx"].get(answer[i], self.vocab["word2idx"]["<unk>"])

        self.data["val"]["inputs"] = np.array(inputs)
        self.data["val"]["questions"] = np.array(questions)
        self.data["val"]["answers"] = np.array(answers)


def bAbI_data_load(path, END=["<end>"]):
    inputs = []
    questions = []
    answers = []

    inputs_len = []
    inputs_sent_len = []
    questions_len = []
    answers_len = []

    for d in open(path):
        index = d.split(" ")[0]
        if index == "1":
            fact = []
        if "?" in d:
            temp = d.split("\t")
            q = temp[0].strip().replace("?", "").split(" ")[1:] + ["?"]
            a = temp[1].split() + END
            fact_copied = deepcopy(fact)
            inputs.append(fact_copied)
            questions.append(q)
            answers.append(a)

            inputs_len.append(len(fact_copied))
            inputs_sent_len.append([len(s) for s in fact_copied])
            questions_len.append(len(q))
            answers_len.append(len(a))
        else:
            tokens = d.replace(".", "").replace("\n", "").split(" ")[1:] + END
            fact.append(tokens)
    return [inputs, questions, answers], [inputs_len, inputs_sent_len, questions_len, answers_len]


# In[4]:


train_data = DataLoader(path="qa5_three-arg-relations_train.txt", is_training=True)
test_data = DataLoader(
    path="qa5_three-arg-relations_test.txt",
    is_training=False,
    vocab=train_data.vocab,
    params=train_data.params,
)


# In[5]:


START = train_data.params["<start>"]
END = train_data.params["<end>"]


# In[6]:


def hop_forward(
    question, memory_o, memory_i, response_proj, inputs_len, questions_len, is_training
):
    match = tf.matmul(question, memory_i, transpose_b=True)
    match = pre_softmax_masking(match, inputs_len)
    match = tf.nn.softmax(match)
    match = post_softmax_masking(match, questions_len)
    response = tf.matmul(match, memory_o)
    return response_proj(tf.concat([response, question], -1))


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
    start = tf.to_int32(tf.fill([batch_size, 1], START))
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
    return tf.convert_to_tensor(np.transpose(encoding))


def input_mem(x, vocab_size, max_sent_len, is_training):
    x = embed_seq(x, vocab_size)
    x = tf.layers.dropout(x, dropout_rate, training=is_training)
    pos = position_encoding(max_sent_len, size_layer)
    x = tf.reduce_sum(x * pos, 2)
    return x


def quest_mem(x, vocab_size, max_quest_len, is_training):
    x = embed_seq(x, vocab_size)
    x = tf.layers.dropout(x, dropout_rate, training=is_training)
    pos = position_encoding(max_quest_len, size_layer)
    return x * pos


# In[7]:


class QA:
    def __init__(self, vocab_size):
        self.questions = tf.placeholder(tf.int32, [None, None])
        self.inputs = tf.placeholder(tf.int32, [None, None, None])
        self.questions_len = tf.placeholder(tf.int32, [None])
        self.inputs_len = tf.placeholder(tf.int32, [None])
        self.answers_len = tf.placeholder(tf.int32, [None])
        self.answers = tf.placeholder(tf.int32, [None, None])
        self.training = tf.placeholder(tf.bool)
        max_sent_len = train_data.params["max_sent_len"]
        max_quest_len = train_data.params["max_quest_len"]
        max_answer_len = train_data.params["max_answer_len"]

        lookup_table = tf.get_variable("lookup_table", [vocab_size, size_layer], tf.float32)
        lookup_table = tf.concat((tf.zeros([1, size_layer]), lookup_table[1:, :]), axis=0)

        with tf.variable_scope("questions"):
            question = quest_mem(self.questions, vocab_size, max_quest_len, self.training)

        with tf.variable_scope("memory_o"):
            memory_o = input_mem(self.inputs, vocab_size, max_sent_len, self.training)

        with tf.variable_scope("memory_i"):
            memory_i = input_mem(self.inputs, vocab_size, max_sent_len, self.training)

        with tf.variable_scope("interaction"):
            response_proj = tf.layers.Dense(size_layer)
            for _ in range(n_hops):
                answer = hop_forward(
                    question,
                    memory_o,
                    memory_i,
                    response_proj,
                    self.inputs_len,
                    self.questions_len,
                    self.training,
                )
                question = answer

        with tf.variable_scope("memory_o", reuse=True):
            embedding = tf.get_variable("lookup_table")
        cell = tf.nn.rnn_cell.LSTMCell(size_layer)
        vocab_proj = tf.layers.Dense(vocab_size)
        state_proj = tf.layers.Dense(size_layer)
        init_state = state_proj(tf.layers.flatten(answer))
        init_state = tf.layers.dropout(init_state, dropout_rate, training=self.training)
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tf.nn.embedding_lookup(embedding, shift_right(self.answers)),
            sequence_length=tf.to_int32(self.answers_len),
        )
        encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=init_state, h=init_state)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell, helper=helper, initial_state=encoder_state, output_layer=vocab_proj
        )
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, maximum_iterations=tf.shape(self.inputs)[1]
        )
        self.outputs = decoder_output.rnn_output

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding,
            start_tokens=tf.tile(tf.constant([START], dtype=tf.int32), [tf.shape(self.inputs)[0]]),
            end_token=END,
        )
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell, helper=helper, initial_state=encoder_state, output_layer=vocab_proj
        )
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, maximum_iterations=max_answer_len
        )
        self.logits = decoder_output.sample_id
        correct_pred = tf.equal(self.logits[:, 0], self.answers[:, 0])
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.cost = tf.reduce_mean(
            tf.contrib.seq2seq.sequence_loss(
                logits=self.outputs,
                targets=self.answers,
                weights=tf.ones_like(self.answers, tf.float32),
            )
        )
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = QA(train_data.params["vocab_size"])
sess.run(tf.global_variables_initializer())


# In[9]:


batching = (train_data.data["val"]["inputs"].shape[0] // batch_size) * batch_size


# In[10]:


for i in range(epoch):
    total_cost, total_acc = 0, 0
    for k in range(0, batching, batch_size):
        batch_questions = train_data.data["val"]["questions"][k : k + batch_size]
        batch_inputs = train_data.data["val"]["inputs"][k : k + batch_size]
        batch_inputs_len = train_data.data["len"]["inputs_len"][k : k + batch_size]
        batch_questions_len = train_data.data["len"]["questions_len"][k : k + batch_size]
        batch_answers_len = train_data.data["len"]["answers_len"][k : k + batch_size]
        batch_answers = train_data.data["val"]["answers"][k : k + batch_size]
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={
                model.questions: batch_questions,
                model.inputs: batch_inputs,
                model.inputs_len: batch_inputs_len,
                model.questions_len: batch_questions_len,
                model.answers_len: batch_answers_len,
                model.answers: batch_answers,
                model.training: True,
            },
        )
        total_cost += cost
        total_acc += acc
    total_cost /= train_data.data["val"]["inputs"].shape[0] // batch_size
    total_acc /= train_data.data["val"]["inputs"].shape[0] // batch_size
    print("epoch %d, avg cost %f, avg acc %f" % (i + 1, total_cost, total_acc))


# In[11]:


testing_size = 32
batch_questions = test_data.data["val"]["questions"][:testing_size]
batch_inputs = test_data.data["val"]["inputs"][:testing_size]
batch_inputs_len = test_data.data["len"]["inputs_len"][:testing_size]
batch_questions_len = test_data.data["len"]["questions_len"][:testing_size]
batch_answers_len = test_data.data["len"]["answers_len"][:testing_size]
batch_answers = test_data.data["val"]["answers"][:testing_size]
logits = sess.run(
    model.logits,
    feed_dict={
        model.questions: batch_questions,
        model.inputs: batch_inputs,
        model.inputs_len: batch_inputs_len,
        model.questions_len: batch_questions_len,
        model.answers_len: batch_answers_len,
        model.training: False,
    },
)


# In[12]:


for i in range(testing_size):
    print("QUESTION:", " ".join([train_data.vocab["idx2word"][k] for k in batch_questions[i]]))
    print("REAL:", train_data.vocab["idx2word"][batch_answers[i, 0]])
    print("PREDICT:", train_data.vocab["idx2word"][logits[i, 0]], "\n")


# In[ ]:
