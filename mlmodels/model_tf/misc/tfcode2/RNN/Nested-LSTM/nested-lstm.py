#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import tensorflow as tf
from sklearn.cross_validation import train_test_split
from tensorflow.python.framework import constant_op, dtypes
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops, init_ops, math_ops, nn_ops, rnn_cell_impl
from tensorflow.python.platform import tf_logging as logging

from utils import *

# In[2]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[3]:


ONEHOT = np.zeros((len(trainset.data), len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)), trainset.target] = 1.0
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(
    trainset.data, trainset.target, ONEHOT, test_size=0.2
)


# In[4]:


concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[5]:


GO = dictionary["GO"]
PAD = dictionary["PAD"]
EOS = dictionary["EOS"]
UNK = dictionary["UNK"]


# In[6]:


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class NLSTMCell(rnn_cell_impl.RNNCell):
    def __init__(
        self,
        num_units,
        depth,
        forget_bias=1.0,
        state_is_tuple=True,
        use_peepholes=True,
        activation=None,
        gate_activation=None,
        cell_activation=None,
        initializer=None,
        input_gate_initializer=None,
        use_bias=True,
        reuse=None,
        name=None,
    ):

        super(NLSTMCell, self).__init__(_reuse=reuse, name=name)
        if not state_is_tuple:
            logging.warn(
                "%s: Using a concatenated state is slower and will soon be "
                "deprecated.  Use state_is_tuple=True.",
                self,
            )

        self.input_spec = base_layer.InputSpec(ndim=2)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._use_peepholes = use_peepholes
        self._depth = depth
        self._activation = activation or math_ops.tanh
        self._gate_activation = gate_activation or math_ops.sigmoid
        self._cell_activation = cell_activation or array_ops.identity
        self._initializer = initializer or init_ops.orthogonal_initializer()
        self._input_gate_initializer = (
            input_gate_initializer or init_ops.glorot_normal_initializer()
        )
        self._use_bias = use_bias
        self._kernels = None
        self._biases = None
        self.built = False

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple([self._num_units] * (self.depth + 1))
        else:
            return self._num_units * (self.depth + 1)

    @property
    def output_size(self):
        return self._num_units

    @property
    def depth(self):
        return self._depth

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernels = []
        if self._use_bias:
            self._biases = []

        if self._use_peepholes:
            self._peep_kernels = []
        for i in range(self.depth):
            if i == 0:
                input_kernel = self.add_variable(
                    "input_gate_kernel",
                    shape=[input_depth, 4 * self._num_units],
                    initializer=self._input_gate_initializer,
                )
                hidden_kernel = self.add_variable(
                    "hidden_gate_kernel",
                    shape=[h_depth, 4 * self._num_units],
                    initializer=self._initializer,
                )
                kernel = tf.concat([input_kernel, hidden_kernel], axis=0, name="kernel_0")
                self._kernels.append(kernel)
            else:
                self._kernels.append(
                    self.add_variable(
                        "kernel_{}".format(i),
                        shape=[2 * h_depth, 4 * self._num_units],
                        initializer=self._initializer,
                    )
                )
            if self._use_bias:
                self._biases.append(
                    self.add_variable(
                        "bias_{}".format(i),
                        shape=[4 * self._num_units],
                        initializer=init_ops.zeros_initializer(dtype=self.dtype),
                    )
                )
            if self._use_peepholes:
                self._peep_kernels.append(
                    self.add_variable(
                        "peep_kernel_{}".format(i),
                        shape=[h_depth, 3 * self._num_units],
                        initializer=self._initializer,
                    )
                )

        self.built = True

    def _recurrence(self, inputs, hidden_state, cell_states, depth):

        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        c = cell_states[depth]
        h = hidden_state

        gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernels[depth])
        if self._use_bias:
            gate_inputs = nn_ops.bias_add(gate_inputs, self._biases[depth])
        if self._use_peepholes:
            peep_gate_inputs = math_ops.matmul(c, self._peep_kernels[depth])
        i_peep, f_peep, o_peep = array_ops.split(
            value=peep_gate_inputs, num_or_size_splits=3, axis=one
        )

        i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)
        if self._use_peepholes:
            i += i_peep
            f += f_peep
            o += o_peep

        if self._use_peepholes:
            peep_gate_inputs = math_ops.matmul(c, self._peep_kernels[depth])
            i_peep, f_peep, o_peep = array_ops.split(
                value=peep_gate_inputs, num_or_size_splits=3, axis=one
            )
            i += i_peep
            f += f_peep
            o += o_peep

        add = math_ops.add
        multiply = math_ops.multiply

        if self._use_bias:
            forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
            f = add(f, forget_bias_tensor)

        inner_hidden = multiply(c, self._gate_activation(f))

        if depth == 0:
            inner_input = multiply(self._gate_activation(i), self._cell_activation(j))
        else:
            inner_input = multiply(self._gate_activation(i), self._activation(j))

        if depth == (self.depth - 1):
            new_c = add(inner_hidden, inner_input)
            new_cs = [new_c]
        else:
            new_c, new_cs = self._recurrence(
                inputs=inner_input,
                hidden_state=inner_hidden,
                cell_states=cell_states,
                depth=depth + 1,
            )
        new_h = multiply(self._activation(new_c), self._gate_activation(o))
        new_cs = [new_h] + new_cs
        return new_h, new_cs

    def call(self, inputs, state):
        if not self._state_is_tuple:
            states = array_ops.split(state, self.depth + 1, axis=1)
        else:
            states = state
        hidden_state = states[0]
        cell_states = states[1:]
        outputs, next_state = self._recurrence(inputs, hidden_state, cell_states, 0)
        if self._state_is_tuple:
            next_state = tuple(next_state)
        else:
            next_state = array_ops.concat(next_state, axis=1)
        return outputs, next_state


# In[7]:


class Model:
    def __init__(
        self,
        size_layer,
        embedded_size,
        dict_size,
        dimension_output,
        learning_rate,
        batch_size,
        timestamp,
        depth=1,
    ):
        self.X = tf.placeholder(tf.int32, [batch_size, maxlen])
        self.Y = tf.placeholder(tf.float32, [batch_size, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        cell = NLSTMCell(size_layer, depth)
        init_state = cell.zero_state(batch_size, dtype=dtypes.float32)
        state = init_state
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(timestamp):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                out, state = cell(encoder_embedded[:, time_step, :], state)
                outputs.append(out)
        outputs = tf.reshape(tf.concat(outputs, axis=1), [batch_size, timestamp, size_layer])
        W = tf.get_variable(
            "w", shape=(size_layer, dimension_output), initializer=tf.orthogonal_initializer()
        )
        b = tf.get_variable("b", shape=(dimension_output), initializer=tf.zeros_initializer())
        self.logits = tf.matmul(outputs[:, -1], W) + b
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[8]:


size_layer = 64
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128


# In[9]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    size_layer,
    embedded_size,
    vocabulary_size + 4,
    dimension_output,
    learning_rate,
    batch_size,
    maxlen,
)
sess.run(tf.global_variables_initializer())


# In[10]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:%d\n" % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (len(train_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(train_X[i : i + batch_size], dictionary, maxlen)
        acc, loss, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: train_onehot[i : i + batch_size]},
        )
        train_loss += loss
        train_acc += acc

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(test_X[i : i + batch_size], dictionary, maxlen)
        acc, loss = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.X: batch_x, model.Y: test_onehot[i : i + batch_size]},
        )
        test_loss += loss
        test_acc += acc

    train_loss /= len(train_X) // batch_size
    train_acc /= len(train_X) // batch_size
    test_loss /= len(test_X) // batch_size
    test_acc /= len(test_X) // batch_size

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
