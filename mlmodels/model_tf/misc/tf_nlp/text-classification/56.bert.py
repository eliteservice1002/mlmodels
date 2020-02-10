#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

import bert_model as modeling
from utils import *

# In[2]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[3]:


concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[4]:


GO = dictionary["GO"]
PAD = dictionary["PAD"]
EOS = dictionary["EOS"]
UNK = dictionary["UNK"]


# In[5]:


size_layer = 128
num_layers = 2
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128


# In[6]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

bert_config = modeling.BertConfig(
    vocab_size=len(dictionary),
    hidden_size=size_layer,
    num_hidden_layers=num_layers,
    num_attention_heads=size_layer // 4,
    intermediate_size=size_layer * 2,
)

input_ids = tf.placeholder(tf.int32, [None, maxlen])
input_mask = tf.placeholder(tf.int32, [None, maxlen])
segment_ids = tf.placeholder(tf.int32, [None, maxlen])
label_ids = tf.placeholder(tf.int32, [None])
is_training = tf.placeholder(tf.bool)


# In[7]:


def create_model(
    bert_config,
    is_training,
    input_ids,
    input_mask,
    segment_ids,
    labels,
    num_labels,
    use_one_hot_embeddings,
    reuse_flag=False,
):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    with tf.variable_scope("weights", reuse=reuse_flag):
        output_weights = tf.get_variable(
            "output_weights",
            [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer()
        )

    with tf.variable_scope("loss"):

        def apply_dropout_last_layer(output_layer):
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            return output_layer

        def not_apply_dropout(output_layer):
            return output_layer

        output_layer = tf.cond(
            is_training,
            lambda: apply_dropout_last_layer(output_layer),
            lambda: not_apply_dropout(output_layer),
        )
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        print(
            "output_layer:",
            output_layer.shape,
            ", output_weights:",
            output_weights.shape,
            ", logits:",
            logits.shape,
        )

        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)
        correct_pred = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return loss, logits, probabilities, model, accuracy


# In[8]:


use_one_hot_embeddings = False
loss, logits, probabilities, model, accuracy = create_model(
    bert_config,
    is_training,
    input_ids,
    input_mask,
    segment_ids,
    label_ids,
    dimension_output,
    use_one_hot_embeddings,
)
global_step = tf.Variable(0, trainable=False, name="Global_Step")
optimizer = tf.contrib.layers.optimize_loss(
    loss, global_step=global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=3.0
)


# In[9]:


sess.run(tf.global_variables_initializer())


# In[10]:


vectors = str_idx(trainset.data, dictionary, maxlen)
train_X, test_X, train_Y, test_Y = train_test_split(vectors, trainset.target, test_size=0.2)


# In[11]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0

while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:%d\n" % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    pbar = tqdm(range(0, len(train_X), batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x = train_X[i : min(i + batch_size, train_X.shape[0])]
        batch_y = train_Y[i : min(i + batch_size, train_X.shape[0])]
        np_mask = np.ones((len(batch_x), maxlen), dtype=np.int32)
        np_segment = np.ones((len(batch_x), maxlen), dtype=np.int32)
        acc, cost, _ = sess.run(
            [accuracy, loss, optimizer],
            feed_dict={
                input_ids: batch_x,
                label_ids: batch_y,
                input_mask: np_mask,
                segment_ids: np_segment,
                is_training: True,
            },
        )
        assert not np.isnan(cost)
        train_loss += cost
        train_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    pbar = tqdm(range(0, len(test_X), batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        np_mask = np.ones((len(batch_x), maxlen), dtype=np.int32)
        np_segment = np.ones((len(batch_x), maxlen), dtype=np.int32)
        acc, cost = sess.run(
            [accuracy, loss],
            feed_dict={
                input_ids: batch_x,
                label_ids: batch_y,
                input_mask: np_mask,
                segment_ids: np_segment,
                is_training: False,
            },
        )
        test_loss += cost
        test_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    train_loss /= len(train_X) / batch_size
    train_acc /= len(train_X) / batch_size
    test_loss /= len(test_X) / batch_size
    test_acc /= len(test_X) / batch_size

    if test_acc > CURRENT_ACC:
        print("epoch: %d, pass acc: %f, current acc: %f" % (EPOCH, CURRENT_ACC, test_acc))
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
    else:
        CURRENT_CHECKPOINT += 1

    print("time taken:", time.time() - lasttime)
    print(
        "epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n"
        % (EPOCH, train_loss, train_acc, test_loss, test_acc)
    )
    EPOCH += 1


# In[12]:


real_Y, predict_Y = [], []

pbar = tqdm(range(0, len(test_X), batch_size), desc="validation minibatch loop")
for i in pbar:
    batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
    batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
    np_mask = np.ones((len(batch_x), maxlen), dtype=np.int32)
    np_segment = np.ones((len(batch_x), maxlen), dtype=np.int32)
    predict_Y += np.argmax(
        sess.run(
            logits,
            feed_dict={
                input_ids: batch_x,
                label_ids: batch_y,
                input_mask: np_mask,
                segment_ids: np_segment,
                is_training: False,
            },
        ),
        1,
    ).tolist()
    real_Y += batch_y


# In[13]:


print(metrics.classification_report(real_Y, predict_Y, target_names=["negative", "positive"]))


# In[ ]:
