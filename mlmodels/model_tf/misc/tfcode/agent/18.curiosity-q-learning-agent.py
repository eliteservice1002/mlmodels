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
    LAYER_SIZE = 500
    OUTPUT_SIZE = 3
    EPSILON = 0.5
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    MEMORIES = deque()
    COPY = 1000
    T_COPY = 0
    MEMORY_SIZE = 300

    def __init__(self, state_size, window_size, trend, skip):
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        tf.reset_default_graph()
        self.X = tf.placeholder(tf.float32, (None, self.state_size))
        self.Y = tf.placeholder(tf.float32, (None, self.state_size))
        self.ACTION = tf.placeholder(tf.float32, (None))
        self.REWARD = tf.placeholder(tf.float32, (None))
        self.batch_size = tf.shape(self.ACTION)[0]

        with tf.variable_scope("curiosity_model"):
            action = tf.reshape(self.ACTION, (-1, 1))
            state_action = tf.concat([self.X, action], axis=1)
            save_state = tf.identity(self.Y)

            feed = tf.layers.dense(state_action, 32, activation=tf.nn.relu)
            self.curiosity_logits = tf.layers.dense(feed, self.state_size)
            self.curiosity_cost = tf.reduce_sum(
                tf.square(save_state - self.curiosity_logits), axis=1
            )

            self.curiosity_optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE).minimize(
                tf.reduce_mean(self.curiosity_cost)
            )

        total_reward = tf.add(self.curiosity_cost, self.REWARD)

        with tf.variable_scope("q_model"):
            with tf.variable_scope("eval_net"):
                x_action = tf.layers.dense(self.X, 128, tf.nn.relu)
                self.logits = tf.layers.dense(x_action, self.OUTPUT_SIZE)

            with tf.variable_scope("target_net"):
                y_action = tf.layers.dense(self.Y, 128, tf.nn.relu)
                y_q = tf.layers.dense(y_action, self.OUTPUT_SIZE)

            q_target = total_reward + self.GAMMA * tf.reduce_max(y_q, axis=1)
            action = tf.cast(self.ACTION, tf.int32)
            action_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), action], axis=1)
            q = tf.gather_nd(params=self.logits, indices=action_indices)
            self.cost = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE).minimize(
                self.cost,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q_model/eval_net"),
            )

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_model/target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_model/eval_net")
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def _memorize(self, state, action, reward, new_state, done):
        self.MEMORIES.append((state, action, reward, new_state, done))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array(res)

    def predict(self, inputs):
        return self.sess.run(self.logits, feed_dict={self.X: inputs})

    def get_predicted_action(self, sequence):
        prediction = self.predict(np.array(sequence))[0]
        return np.argmax(prediction)

    def _select_action(self, state):
        if np.random.rand() < self.EPSILON:
            action = np.random.randint(self.OUTPUT_SIZE)
        else:
            action = self.get_predicted_action([state])
        return action

    def _construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        actions = np.array([a[1] for a in replay])
        rewards = np.array([a[2] for a in replay])
        new_states = np.array([a[3] for a in replay])
        if (self.T_COPY + 1) % self.COPY == 0:
            self.sess.run(self.target_replace_op)

        cost, _ = self.sess.run(
            [self.cost, self.optimizer],
            feed_dict={
                self.X: states,
                self.Y: new_states,
                self.ACTION: actions,
                self.REWARD: rewards,
            },
        )

        if (self.T_COPY + 1) % self.COPY == 0:
            self.sess.run(
                self.curiosity_optimizer,
                feed_dict={
                    self.X: states,
                    self.Y: new_states,
                    self.ACTION: actions,
                    self.REWARD: rewards,
                },
            )
        return cost

    def buy(self, initial_money):
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        state = self.get_state(0)
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self._select_action(state)
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

            state = next_state
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest

    def train(self, iterations, checkpoint, initial_money):
        for i in range(iterations):
            total_profit = 0
            inventory = []
            state = self.get_state(0)
            starting_money = initial_money
            for t in range(0, len(self.trend) - 1, self.skip):

                action = self._select_action(state)
                next_state = self.get_state(t + 1)

                if action == 1 and starting_money >= self.trend[t]:
                    inventory.append(self.trend[t])
                    starting_money -= self.trend[t]

                elif action == 2 and len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    total_profit += self.trend[t] - bought_price
                    starting_money += self.trend[t]

                invest = (starting_money - initial_money) / initial_money

                self._memorize(state, action, invest, next_state, starting_money < initial_money)
                batch_size = min(len(self.MEMORIES), self.BATCH_SIZE)
                state = next_state
                replay = random.sample(self.MEMORIES, batch_size)
                cost = self._construct_memories(replay)
                self.T_COPY += 1
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


# In[9]:


states_buy, states_sell, total_gains, invest = agent.buy(initial_money=initial_money)


# In[10]:


fig = plt.figure(figsize=(15, 5))
plt.plot(close, color="r", lw=2.0)
plt.plot(close, "^", markersize=10, color="m", label="buying signal", markevery=states_buy)
plt.plot(close, "v", markersize=10, color="k", label="selling signal", markevery=states_sell)
plt.title("total gains %f, total investment %f%%" % (total_gains, invest))
plt.legend()
plt.show()


# In[ ]:
