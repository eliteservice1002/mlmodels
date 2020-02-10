"""
LSTM Attention

"""

import numpy as np
import tensorflow as tf
import os

from tensorflow import keras

K = keras.backend

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import RepeatVector, Concatenate, Activation
from tensorflow.python.keras.layers import Reshape, Input, Dense, Dot, LSTM
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.models import load_model as keras_load_model

import matplotlib.pyplot as plt
import pylab


# Sometimes helpful to implement own softmax activation function to
# better manage calculations along specific axes.
def softmax_activation(x):
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e / s


class AttentionModel(object):

    def __init__(self, x, y,
                 layer_1_rnn_units,
                 attn_dense_nodes=0,
                 epochs=100,
                 batch_size=128,
                 shared_attention_layer=True,
                 chg_yield=False,
                 float_type='float32',
                 regularization=(0.00001, '00001'),
                 window=52,
                 predict=1):
        K.clear_session()
        tf.reset_default_graph()
        self.set_learning(True)

        # Scientific computing uses 'float64' but
        # machine learning works much faster with 'float32'.
        self.float_type = float_type
        K.set_floatx(self.float_type)

        # Capture inputs to instance variables.
        self.x = x
        self.y = y
        self.epochs = epochs
        self.batch_size = batch_size
        self.shared_attention_layer = shared_attention_layer

        self.layer_1_rnn_units = layer_1_rnn_units
        self.layer_2_rnn_units = self.layer_1_rnn_units
        self.attn_dense_nodes = attn_dense_nodes

        self.num_obs = self.x.shape[0]
        self.input_len = self.x.shape[1]
        self.input_dims = self.x.shape[2]
        self.num_outputs = self.y.shape[1]

        self.regularization = regularization[0]

        assert self.x.shape[0] == self.y.shape[0]

        # Set the directory structure.
        self.model_dir = f'models//window_{window}_predict_{predict}//'

        self.model_name = f'{"yield_changes" if chg_yield==True else "yield_levels"}//' \
                          f'model_{layer_1_rnn_units}_rnn_{attn_dense_nodes}_dense_attn_' \
                          f'{epochs}_epochs_' \
                          f'{batch_size}_batch_' \
                          f'{"shared_attention" if shared_attention_layer else""}_' \
                          f'{"change_yield" if chg_yield else "level_yield"}_' \
                          f'{regularization[1]}_reg'

        # Activation function for the attention mechanism dense layer(s).
        self.attn_dense_activation = 'selu'
        self.attn_dense_initializer = 'lecun_normal'
        
    def delete_model(self):
        try:
            os.remove(f'{self.model_dir}{self.model_name}.h5')
        except Exception as e:
            print(e)

    def load_model(self):
        try:
            self.model = keras_load_model(f'{self.model_dir}{self.model_name}.h5',
                                          custom_objects={'softmax_activation': softmax_activation})
        except Exception as e:
            print(e)
            return False
        return True

    def save_model(self):
        try:
            self.model.save(f'{self.model_dir}{self.model_name}.h5')
        except Exception as e:
            print(e)
            return False
        return True

    def set_learning(self, learning):
        if learning:
            self.is_learning_phase = 1
            K.set_learning_phase(self.is_learning_phase)
            tf.keras.backend.set_learning_phase(True)
        else:
            self.is_learning_phase = 0
            K.set_learning_phase(self.is_learning_phase)
            tf.keras.backend.set_learning_phase(False)

    # Method that constructs shared layers. A shared layer means its learned parameters
    # are the same no matter where the layer is used in the neural network.
    #
    def make_shared_layers(self):
        if self.regularization > 0.:
            self.kernel_reg = regularizers.l2(self.regularization)
            self.bias_reg = regularizers.l2(self.regularization)
            self.recurrent_reg = regularizers.l2(self.regularization)
            self.recurrent_dropout = 0.1
        else:
            self.kernel_reg = self.bias_reg = self.recurrent_reg = None
            self.recurrent_dropout = 0.0

        if self.shared_attention_layer:

            # This is an optional intermediate dense layer in the attention network.
            # If it is not present, the attention mechanism goes straight from inputs to weights.
            if self.attn_dense_nodes > 0:
                self.attn_middle_dense_layer = Dense(self.attn_dense_nodes,
                                                     kernel_regularizer=self.kernel_reg,
                                                     bias_regularizer=self.bias_reg,
                                                     activation=self.attn_dense_activation,
                                                     kernel_initializer=self.attn_dense_initializer,
                                                     name='attention_mid_dense_shared')

            # This is the layer in the attention mechanism that gives the attention weights.
            self.attention_final_dense_layer = Dense(1,
                                                     kernel_regularizer=self.kernel_reg,
                                                     bias_regularizer=self.bias_reg,
                                                     activation=self.attn_dense_activation,
                                                     kernel_initializer=self.attn_dense_initializer,
                                                     name='attention_final_dense_shared')

        # Output-level LSTM cell.
        self.layer_2_LSTM_cell = LSTM(self.layer_2_rnn_units,
                                      kernel_regularizer=self.kernel_reg,
                                      recurrent_regularizer=self.recurrent_reg,
                                      bias_regularizer=self.bias_reg,
                                      recurrent_dropout=self.recurrent_dropout,
                                      return_state=True,
                                      name='layer_2_LSTM')

        # Final output (i.e., the prediction).
        self.dense_output = Dense(1,
                                  kernel_regularizer=self.kernel_reg,
                                  bias_regularizer=self.bias_reg,
                                  activation='linear',
                                  name='dense_output')

    # Builds the neural network. An LSTM+attention model doesn't need this much code.
    # This method is long because it sets lots of layer parameters and because
    # it handles four contingencies: (1) whether the attention mechanism is
    # always the same or is different for every prediction node, and (2) whether or
    # not the attention mechanism has an intermediate dense layer.
    #
    def build_attention_rnn(self):
        self.make_shared_layers()

        inputs = Input(shape=(self.input_len, self.input_dims), dtype=self.float_type)

        X = LSTM(self.layer_1_rnn_units,
                 kernel_regularizer=self.kernel_reg,
                 recurrent_regularizer=self.recurrent_reg,
                 bias_regularizer=self.bias_reg,
                 recurrent_dropout=self.recurrent_dropout,
                 return_sequences=True)(inputs)

        X = Reshape((self.input_len, self.layer_2_rnn_units))(X)

        h_start = Input(shape=(self.layer_2_rnn_units,), name='h_start')
        c_start = Input(shape=(self.layer_2_rnn_units,), name='c_start')
        h_prev = h_start
        c_prev = c_start

        outputs = list()

        # This section constructs the attention mechanism and the output-level LSTM
        # layer that leads to the predictions.
        #
        # There is an extra LSTM cell that is not attached to any prediction but
        # which begins the output-level RNN sequence. This avoids sending in a bunch
        # of zero values to the first usage of the attention mechanism.
        #
        # One way to avoid this extra LSTM cell might be to set the LSTM intial state
        # tensors "h_start" and "c_start" as trainable (instead of zeros).
        #
        for t in range(self.num_outputs + 1):
            h_prev_repeat = RepeatVector(self.input_len)(h_prev)
            joined = Concatenate(axis=-1)([X, h_prev_repeat])

            if self.attn_dense_nodes > 0:
                if self.shared_attention_layer:
                    joined = self.attn_middle_dense_layer(joined)
                else:
                    joined = Dense(self.attn_dense_nodes,
                                   kernel_regularizer=self.kernel_reg,
                                   bias_regularizer=self.bias_reg,
                                   activation=self.attn_dense_activation,
                                   kernel_initializer=self.attn_dense_initializer,
                                   name=f'attention_mid_dense_{t}')(joined)

            if self.shared_attention_layer:
                e_vals = self.attention_final_dense_layer(joined)
            else:
                e_vals = Dense(1,
                               kernel_regularizer=self.kernel_reg,
                               bias_regularizer=self.bias_reg,
                               activation=self.attn_dense_activation,
                               kernel_initializer=self.attn_dense_initializer,
                               name=f'attention_final_dense_{t}')(joined)

            alphas = Activation(softmax_activation, name=f'attention_softmax_{t}')(e_vals)
            attentions = Dot(axes=1)([alphas, X])

            h_prev, _, c_prev = self.layer_2_LSTM_cell(attentions, initial_state=[h_prev, c_prev])

            if t > 0:
                out = self.dense_output(h_prev)
                outputs.append(out)

        self.model = Model(inputs=[inputs, h_start, c_start], outputs=outputs)
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        print(self.model.summary())

    def fit_model(self):
        self.set_learning(True)

        h_start = np.zeros((self.num_obs, self.layer_2_rnn_units))
        c_start = np.zeros((self.num_obs, self.layer_2_rnn_units))

        y_split = np.split(self.y, indices_or_sections=self.num_outputs, axis=1)

        self.model.fit([self.x, h_start, c_start],
                       y_split,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       shuffle=True,
                       verbose=2,
                       validation_split=0.1)

    def calculate_attentions(self, x_data):
        self.set_learning(False)

        softmax_layer_names = [f'attention_softmax_{t}' for t in range(self.num_outputs + 1)]
        softmax_layers = list()

        for i, layer_name in enumerate(softmax_layer_names):
            if i == 0:
                continue
            intermediate_layer = Model(inputs=self.model.input,
                                       outputs=self.model.get_layer(layer_name).output)
            softmax_layers.append(intermediate_layer)

        num_obs = x_data.shape[0]
        attention_map = np.zeros((num_obs, self.num_outputs, self.input_len))

        h_start = np.zeros((1, self.layer_2_rnn_units))
        c_start = np.zeros((1, self.layer_2_rnn_units))

        for t in range(num_obs):
            print(t)
            for l_num, layer in enumerate(softmax_layers):
                softmax_results = layer.predict([np.expand_dims(x_data[t], axis=0),
                                                 h_start,
                                                 c_start])
                softmax_results = softmax_results[0, :, 0]
                attention_map[t, l_num, :] = softmax_results

        return attention_map

    def heatmap(self, data, title_supplement=None):
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.labelsize'] = 22
        plt.rcParams['axes.titlesize'] = 22
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['axes.titlepad'] = 12
        plt.rcParams['axes.edgecolor'] = '#000000'  # '#FD5E0F'

        # Other common color schemes: 'viridis'  'plasma'  'gnuplot'
        color_map = 'inferno'
        pylab.pcolor(data, cmap=color_map, vmin=0.)
        pylab.colorbar()

        num_predictions = data.shape[0]
        num_timesteps = data.shape[1]

        if num_predictions == 4:
            pylab.yticks([0.5, 1.5, 2.5, 3.5], ['t+1', 't+2', 't+3', 't+4'])
            pylab.ylabel('y: t+1 to t+4')

            plt.axhline(y=1., xmin=0.0, xmax=51.0, linewidth=1, color='w')
            plt.axhline(y=2., xmin=0.0, xmax=51.0, linewidth=1, color='w')
            plt.axhline(y=3., xmin=0.0, xmax=51.0, linewidth=1, color='w')

        elif num_predictions == 1:
            pylab.yticks([0.5], ['t+1'])
            pylab.ylabel('y: t+1')

        assert num_timesteps == 52

        pylab.xticks([1.5, 11.5, 21.5, 31.5, 41.5, 51.5],
                     ['t-50', 't-40', 't-30', 't-20', 't-10', 't'])
        pylab.xlabel('x: t-51 to t')

        pylab.title(f'{self.model_name} {title_supplement}')

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        pylab.show()
        
        
        
        



