#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

sns.set()


# In[2]:


df = pd.read_csv("../dataset/GOOG-year.csv")
df.head()


# In[3]:


class Agent:

    LEARNING_RATE = 0.003
    BATCH_SIZE = 32
    LAYER_SIZE = 256
    OUTPUT_SIZE = 3
    EPSILON = 0.5
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    MEMORIES = deque()
    MEMORY_SIZE = 300

    def __init__(self, state_size, window_size, trend, skip):
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        tf.reset_default_graph()
        self.INITIAL_FEATURES = np.zeros((4, self.state_size))
        self.X = tf.placeholder(tf.float32, (None, None, self.state_size))
        self.Y = tf.placeholder(tf.float32, (None, self.OUTPUT_SIZE))
        cell = tf.nn.rnn_cell.LSTMCell(self.LAYER_SIZE, state_is_tuple=False)
        self.hidden_layer = tf.placeholder(tf.float32, (None, 2 * self.LAYER_SIZE))
        self.rnn, self.last_state = tf.nn.dynamic_rnn(
            inputs=self.X, cell=cell, dtype=tf.float32, initial_state=self.hidden_layer
        )
        self.logits = tf.layers.dense(self.rnn[:, -1], self.OUTPUT_SIZE)
        self.cost = tf.reduce_sum(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(
            self.cost
        )
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def _memorize(self, state, action, reward, new_state, dead, rnn_state):
        self.MEMORIES.append((state, action, reward, new_state, dead, rnn_state))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

    def _construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        init_values = np.array([a[-1] for a in replay])
        Q = self.sess.run(self.logits, feed_dict={self.X: states, self.hidden_layer: init_values})
        Q_new = self.sess.run(
            self.logits, feed_dict={self.X: new_states, self.hidden_layer: init_values}
        )
        replay_size = len(replay)
        X = np.empty((replay_size, 4, self.state_size))
        Y = np.empty((replay_size, self.OUTPUT_SIZE))
        INIT_VAL = np.empty((replay_size, 2 * self.LAYER_SIZE))
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, dead_r, rnn_memory = replay[i]
            target = Q[i]
            target[action_r] = reward_r
            if not dead_r:
                target[action_r] += self.GAMMA * np.amax(Q_new[i])
            X[i] = state_r
            Y[i] = target
            INIT_VAL[i] = rnn_memory
        return X, Y, INIT_VAL

    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array(res)

    def buy(self, initial_money):
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        state = self.get_state(0)
        init_value = np.zeros((1, 2 * self.LAYER_SIZE))
        for k in range(self.INITIAL_FEATURES.shape[0]):
            self.INITIAL_FEATURES[k, :] = state
        for t in range(0, len(self.trend) - 1, self.skip):
            action, last_state = self.sess.run(
                [self.logits, self.last_state],
                feed_dict={self.X: [self.INITIAL_FEATURES], self.hidden_layer: init_value},
            )
            action, init_value = np.argmax(action[0]), last_state
            next_state = self.get_state(t + 1)

            if action == 1 and initial_money >= self.trend[t]:
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
                print(
                    "day %d: buy 1 unit at price %f, total balance %f"
                    % (t, self.trend[t], initial_money)
                )

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.trend[t]
                states_sell.append(t)
                try:
                    invest = ((close[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    "day %d, sell 1 unit at price %f, investment %f %%, total balance %f,"
                    % (t, close[t], invest, initial_money)
                )

            new_state = np.append([self.get_state(t + 1)], self.INITIAL_FEATURES[:3, :], axis=0)
            self.INITIAL_FEATURES = new_state
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest

    def train(self, iterations, checkpoint, initial_money):
        for i in range(iterations):
            total_profit = 0
            inventory = []
            state = self.get_state(0)
            starting_money = initial_money
            init_value = np.zeros((1, 2 * self.LAYER_SIZE))
            for k in range(self.INITIAL_FEATURES.shape[0]):
                self.INITIAL_FEATURES[k, :] = state
            for t in range(0, len(self.trend) - 1, self.skip):

                if np.random.rand() < self.EPSILON:
                    action = np.random.randint(self.OUTPUT_SIZE)
                else:
                    action, last_state = self.sess.run(
                        [self.logits, self.last_state],
                        feed_dict={self.X: [self.INITIAL_FEATURES], self.hidden_layer: init_value},
                    )
                    action, init_value = np.argmax(action[0]), last_state

                next_state = self.get_state(t + 1)

                if action == 1 and starting_money >= self.trend[t]:
                    inventory.append(self.trend[t])
                    starting_money -= self.trend[t]

                elif action == 2 and len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    total_profit += self.trend[t] - bought_price
                    starting_money += self.trend[t]

                invest = (starting_money - initial_money) / initial_money
                new_state = np.append([self.get_state(t + 1)], self.INITIAL_FEATURES[:3, :], axis=0)
                self._memorize(
                    self.INITIAL_FEATURES,
                    action,
                    invest,
                    new_state,
                    starting_money < initial_money,
                    init_value[0],
                )
                self.INITIAL_FEATURES = new_state
                batch_size = min(len(self.MEMORIES), self.BATCH_SIZE)
                replay = random.sample(self.MEMORIES, batch_size)
                X, Y, INIT_VAL = self._construct_memories(replay)

                cost, _ = self.sess.run(
                    [self.cost, self.optimizer],
                    feed_dict={self.X: X, self.Y: Y, self.hidden_layer: INIT_VAL},
                )
                self.EPSILON = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * np.exp(
                    -self.DECAY_RATE * i
                )

            if (i + 1) % checkpoint == 0:
                print(
                    "epoch: %d, total rewards: %f.3, cost: %f, total money: %f"
                    % (i + 1, total_profit, cost, starting_money)
                )


# In[4]:


close = df.Close.values.tolist()
initial_money = 10000
window_size = 30
skip = 1
batch_size = 32
agent = Agent(state_size=window_size, window_size=window_size, trend=close, skip=skip)
agent.train(iterations=200, checkpoint=10, initial_money=initial_money)


# In[5]:


states_buy, states_sell, total_gains, invest = agent.buy(initial_money=initial_money)


# In[6]:


fig = plt.figure(figsize=(15, 5))
plt.plot(close, color="r", lw=2.0)
plt.plot(close, "^", markersize=10, color="m", label="buying signal", markevery=states_buy)
plt.plot(close, "v", markersize=10, color="k", label="selling signal", markevery=states_sell)
plt.title("total gains %f, total investment %f%%" % (total_gains, invest))
plt.legend()
plt.show()


# In[ ]:
