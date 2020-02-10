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


class Actor:
    def __init__(self, name, input_size, output_size, size_layer):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, (None, None, input_size))
            self.hidden_layer = tf.placeholder(tf.float32, (None, 2 * size_layer))
            cell = tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)
            self.rnn, self.last_state = tf.nn.dynamic_rnn(
                inputs=self.X, cell=cell, dtype=tf.float32, initial_state=self.hidden_layer
            )
            self.logits = tf.layers.dense(self.rnn[:, -1], output_size)


class Critic:
    def __init__(self, name, input_size, output_size, size_layer, learning_rate):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, (None, None, input_size))
            self.Y = tf.placeholder(tf.float32, (None, output_size))
            self.hidden_layer = tf.placeholder(tf.float32, (None, 2 * size_layer))
            self.REWARD = tf.placeholder(tf.float32, (None, 1))
            feed_critic = tf.layers.dense(self.X, size_layer, activation=tf.nn.relu)
            cell = tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)
            self.rnn, self.last_state = tf.nn.dynamic_rnn(
                inputs=self.X, cell=cell, dtype=tf.float32, initial_state=self.hidden_layer
            )
            feed_critic = (
                tf.layers.dense(self.rnn[:, -1], output_size, activation=tf.nn.relu) + self.Y
            )
            feed_critic = tf.layers.dense(feed_critic, size_layer // 2, activation=tf.nn.relu)
            self.logits = tf.layers.dense(feed_critic, 1)
            self.cost = tf.reduce_mean(tf.square(self.REWARD - self.logits))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


class Agent:

    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    LAYER_SIZE = 256
    OUTPUT_SIZE = 3
    EPSILON = 0.5
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    MEMORIES = deque()
    MEMORY_SIZE = 300
    COPY = 1000
    T_COPY = 0

    def __init__(self, state_size, window_size, trend, skip):
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.INITIAL_FEATURES = np.zeros((4, self.state_size))
        self.skip = skip
        tf.reset_default_graph()
        self.actor = Actor("actor-original", self.state_size, self.OUTPUT_SIZE, self.LAYER_SIZE)
        self.actor_target = Actor(
            "actor-target", self.state_size, self.OUTPUT_SIZE, self.LAYER_SIZE
        )
        self.critic = Critic(
            "critic-original",
            self.state_size,
            self.OUTPUT_SIZE,
            self.LAYER_SIZE,
            self.LEARNING_RATE,
        )
        self.critic_target = Critic(
            "critic-target", self.state_size, self.OUTPUT_SIZE, self.LAYER_SIZE, self.LEARNING_RATE
        )
        self.grad_critic = tf.gradients(self.critic.logits, self.critic.Y)
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.OUTPUT_SIZE])
        weights_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor")
        self.grad_actor = tf.gradients(self.actor.logits, weights_actor, -self.actor_critic_grad)
        grads = zip(self.grad_actor, weights_actor)
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(grads)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def _assign(self, from_name, to_name):
        from_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_name)
        to_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_name)
        for i in range(len(from_w)):
            assign_op = to_w[i].assign(from_w[i])
            self.sess.run(assign_op)

    def _memorize(self, state, action, reward, new_state, dead, rnn_state):
        self.MEMORIES.append((state, action, reward, new_state, dead, rnn_state))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

    def _select_action(self, state):
        if np.random.rand() < self.EPSILON:
            action = np.random.randint(self.OUTPUT_SIZE)
        else:
            prediction = self.sess.run(self.actor.logits, feed_dict={self.actor.X: [state]})[0]
            action = np.argmax(prediction)
        return action

    def _construct_memories_and_train(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        init_values = np.array([a[-1] for a in replay])
        Q = self.sess.run(
            self.actor.logits,
            feed_dict={self.actor.X: states, self.actor.hidden_layer: init_values},
        )
        Q_target = self.sess.run(
            self.actor_target.logits,
            feed_dict={self.actor_target.X: states, self.actor_target.hidden_layer: init_values},
        )
        grads = self.sess.run(
            self.grad_critic,
            feed_dict={
                self.critic.X: states,
                self.critic.Y: Q,
                self.critic.hidden_layer: init_values,
            },
        )[0]
        self.sess.run(
            self.optimizer,
            feed_dict={
                self.actor.X: states,
                self.actor_critic_grad: grads,
                self.actor.hidden_layer: init_values,
            },
        )

        rewards = np.array([a[2] for a in replay]).reshape((-1, 1))
        rewards_target = self.sess.run(
            self.critic_target.logits,
            feed_dict={
                self.critic_target.X: new_states,
                self.critic_target.Y: Q_target,
                self.critic_target.hidden_layer: init_values,
            },
        )
        for i in range(len(replay)):
            if not replay[0][-2]:
                rewards[i] += self.GAMMA * rewards_target[i]
        cost, _ = self.sess.run(
            [self.critic.cost, self.critic.optimizer],
            feed_dict={
                self.critic.X: states,
                self.critic.Y: Q,
                self.critic.REWARD: rewards,
                self.critic.hidden_layer: init_values,
            },
        )
        return cost

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

            if np.random.rand() < self.EPSILON:
                action = np.random.randint(self.OUTPUT_SIZE)
            else:
                action, last_state = self.sess.run(
                    [self.actor.logits, self.actor.last_state],
                    feed_dict={
                        self.actor.X: [self.INITIAL_FEATURES],
                        self.actor.hidden_layer: init_value,
                    },
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
                if (self.T_COPY + 1) % self.COPY == 0:
                    self._assign("actor-original", "actor-target")
                    self._assign("critic-original", "critic-target")

                if np.random.rand() < self.EPSILON:
                    action = np.random.randint(self.OUTPUT_SIZE)
                else:
                    action, last_state = self.sess.run(
                        [self.actor.logits, self.actor.last_state],
                        feed_dict={
                            self.actor.X: [self.INITIAL_FEATURES],
                            self.actor.hidden_layer: init_value,
                        },
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
                batch_size = min(len(self.MEMORIES), self.BATCH_SIZE)
                self.INITIAL_FEATURES = new_state
                replay = random.sample(self.MEMORIES, batch_size)
                cost = self._construct_memories_and_train(replay)
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
