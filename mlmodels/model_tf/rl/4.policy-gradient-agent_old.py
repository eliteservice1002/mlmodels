# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

sns.set()




class Model:
    def __init__(self,
        state_size,
        window_size,
        trend,
        skip,
        iterations,
        initial_reward
    ):
        self.agent = Agent(state_size, window_size, trend, skip)
        self.iterations = iterations
        self.initial_reward = initial_reward


class Agent:
    LEARNING_RATE = 1e-4
    LAYER_SIZE = 256
    GAMMA = 0.9
    OUTPUT_SIZE = 3

    def __init__(self, state_size, window_size, trend, skip):
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend  # history
        self.skip = skip    # time step
        
        
        ### RL models
        self.X = tf.placeholder(tf.float32, (None, self.state_size))
        self.REWARDS = tf.placeholder(tf.float32, (None))
        self.ACTIONS = tf.placeholder(tf.int32, (None))
        feed_forward = tf.layers.dense(self.X, self.LAYER_SIZE, activation = tf.nn.relu)
        self.logits = tf.layers.dense(feed_forward, self.OUTPUT_SIZE, activation = tf.nn.softmax)
        input_y = tf.one_hot(self.ACTIONS, self.OUTPUT_SIZE)
        loglike = tf.log((input_y * (input_y - self.logits) + (1 - input_y) * (input_y + self.logits)) + 1)
        rewards = tf.tile(tf.reshape(self.REWARDS, (-1,1)), [1, self.OUTPUT_SIZE])
        self.cost = -tf.reduce_mean(loglike * (rewards + 1)) 
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    
    def predict(self, inputs):
        # Predict action_i
        return self.sess.run(self.logits, feed_dict={self.X: inputs})


    def get_state(self, t, reward_state=None):
        """
         Action ---> ENV -->  Reward +Change in State== reward state 
         In this particular case, there is no rewaed state
         State is difference in price over window
        
        """
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])
    
    
    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.GAMMA + r[t]
            discounted_r[t] = running_add
        return discounted_r
    
    
    def get_predicted_action(self, sequence):
        prediction = self.predict(np.array(sequence))[0]  #Proba vector for each action
        return np.argmax(prediction)
    
    
    def predict_sequence(self, trend_input, do_action , param=None):
        """
          Generate the states, and get action result intoList
        
          trend_input :  timeseries of state
          do_action   :  action to be done
          param       :  parameters into dict

        """
        # state          = self.get_state(0)
        # starting_reward = param["initial_reward"]   
        
        state = param["initial_state"]

        action_history = []
        ep_history     = []
        total_reward   = 0
        inventory      = [] # passed by reference
     
        for t in range(0, len(trend_input) - 1, self.skip):
            action    = self.get_predicted_action(state)
       
            action_dict = {"t": t, 
                           "action": action,             
            
                           "param":  param,    
                           # { "half_window": agent.half_window},
                           # "starting_reward": starting_reward, 
                           
                           "history"      : trend_input,    #History price
                           "total_reward" : total_reward,
                           "inventory"    : inventory
                          }
       
            d            = do_action( action_dict )  
            
            inventory    = d["inventory"]
            total_reward = d["total_reward"]
            reward_t     = d["reward"]
            reward_state = d.get("reward_state") # None is ok
            
            next_state  = self.get_state(t + 1, reward_state)  # can be None
            action_history.append( d )
            ep_history.append([state, action, reward_t ,next_state])
            
            state = next_state


        ep_history      = np.array(ep_history)
        ep_history[:,2] = agent.discount_rewards(ep_history[:,2])
        return ep_history, action_history
        

    
    def buy(self, initial_money):
        starting_money = initial_money
        states_sell    = []
        states_buy     = []
        inventory      = []
        
        
        state = self.get_state(0)
        for t in range(0, len(self.trend) - 1, self.skip):
            action     = self.get_predicted_action(state)
            
            ###### do_action ########################################
            if action == 1 and initial_money >= self.trend[t] and t < (len(self.trend) - self.half_window):
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f'% (t, self.trend[t], initial_money))
                
                
            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.trend[t]
                states_sell.append(t)
                try:
                    invest = ((close[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, close[t], invest, initial_money)
                )
            ########################################################
            next_state = self.get_state(t + 1)
            state = next_state
            
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest
        
    
    def train(self, iterations, checkpoint, initial_money):
        # Old version
        for i in range(iterations):
            ep_history = []
            total_profit = 0
            inventory = []
            state = self.get_state(0)
            starting_money = initial_money
            
            
            for t in range(0, len(self.trend) - 1, self.skip):
                action = self.get_predicted_action(state)
                next_state = self.get_state(t + 1)
                
                
                ######## do_action ###################################
                if action == 1 and starting_money >= self.trend[t] and t < (len(self.trend) - self.half_window):
                    inventory.append(self.trend[t])
                    starting_money -= close[t]
                
                elif action == 2 and len(inventory):
                    bought_price = inventory.pop(0)
                    total_profit += self.trend[t] - bought_price
                    starting_money += self.trend[t]
                ###################################################
                
                
                ep_history.append([state,action,starting_money,next_state])
                state = next_state
            ep_history = np.array(ep_history)
            ep_history[:,2] = self.discount_rewards(ep_history[:,2])
            cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.X:np.vstack(ep_history[:,0]),
                                                    self.REWARDS:ep_history[:,2],
                                                    self.ACTIONS:ep_history[:,1]})
            if (i+1) % checkpoint == 0:
                print('epoch: %d, total rewards: %f.3, cost: %f, total money: %f'%(i + 1, total_profit, cost,
                                                                                  starting_money))
    
    

def fit(model, df, do_action):
    agent = model.agent
    for i in range(model.iterations):
        ep_history = []
        total_reward = 0
        inventory = [] # passed by reference
        state = agent.get_state(0)
        starting_reward = model.initial_reward
        
        
        for t in range(0, len(agent.trend) - 1, agent.skip):
            action = agent.get_predicted_action(state)
            param = { "half_window"     : agent.half_window,"starting_reward" : starting_reward }
            
            trend_input = agent.trend
            ######## do_action ###################################
            action_dict = {"t": t, 
                           "action": action,             
            
                           "param":  param,    
                           
                           "history": trend_input, 
                           "total_reward": total_reward,
                           "inventory"   : inventory
                          }
                          
            r = do_action(action_dict)
            starting_reward        = r["reward_t"]  #### Be careful
            
            total_reward,inventory = r["total_reward"] ,  r["inventory"]
            reward_state           = r.get("reward_state")
            next_state = agent.get_state(t + 1, reward_state= reward_state)
            ###################################################
            ep_history.append([state, action, starting_reward, next_state])
            state = next_state


        ep_history = np.array(ep_history)
        ep_history[:,2] = agent.discount_rewards(ep_history[:,2])
        cost, _ = agent.sess.run([agent.cost, agent.optimizer], 
                                 feed_dict= {agent.X:np.vstack(ep_history[:,0]),
                                 agent.REWARDS:ep_history[:,2],
                                 agent.ACTIONS:ep_history[:,1]})

        if (i+1) % 10 == 0:
            print('epoch: %d, total rewards: %f.3, cost: %f, total money: %f'%(i + 1, 
            total_reward, cost, starting_reward))

    return agent.sess



def predict(model, sess, df, do_action):
    model.agent.sess = sess
    res = model.agent.predict_sequence(df, do_action_example) #TODO needs an example function to work
    return res




################################################################################################
################################################################################################
#https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
class to_name(object):
  def __init__(self, adict):
    self.__dict__.update(adict)



def do_action_example(action_dict):
    """
        starting_money = initial_money
        states_sell    = []
        states_buy     = []
        inventory      = []
        
        state = self.get_state(0)
        for t in range(0, len(self.trend) - 1, self.skip):
            action     = self.get_predicted_action(state)
            
            ###### do_action ########################################
            if action == 1 and initial_money >= self.trend[t] and t < (len(self.trend) - self.half_window):
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f'% (t, self.trend[t], initial_money))
                
                
            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.trend[t]
                states_sell.append(t)
                try:
                    invest = ((close[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, close[t], invest, initial_money)
                )
            ########################################################
            next_state = self.get_state(t + 1)
            state = next_state
            
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest    
    
    

    
    
    """
    x         = to_name(action_dict)

    ########## Mapping ####################################
    t            = x.t # time step
    action       = x.action  #selectec action
    price        = x.history[x.t]   # current price from history price
    total_reward = x.total_reward
    inventory    = x.inventory    # how much we have in stocks    
    
    
    initial_money = x.current_state_val
    starting_money = initial_money
    
    half_window      = x.param['half_window']
    starting_reward  = x.param['starting_reward']  
    states_sell , states_buy = [], []
    
    #######################################################
    #### Buy
    if action == 1 and starting_reward >= price and t < (len(x.history) - half_window):
        inventory.append(price)
        initial_money  = initial_money - price
        states_buy.appent(t)
        print('day %d: buy 1 unit at price %f, total balance %f'% (t, price, initial_money))
        reward_t = 0

        
    ### Sell    
    elif action == 2 and len(inventory):
        bought_price     = inventory.pop(0)
        initial_money    = initial_money + price
        states_sell.appent(t)       
        
        
        reward_t         = price - bouginitial_money + priceht_price
        total_reward    += reward_t
        starting_reward += price

        try:
                    invest = ((price - bought_price) / bought_price) * 100
        except:
                    invest = 0
 
        print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, price, invest, initial_money)
        )   
    
    ##### Mapping Back #################################
    invest = ((initial_money - starting_money) / starting_money) * 100
    total_gains = initial_money - starting_money
    
    
    d = { "inventory"      : inventory,
          
          "reward_t"       : invest, 
          "total_reward"   : total_gains,
          "reward_state"   : None, 
          
          "states_buy"     : states_buy,
          "states_sell"    : states_sell,
        } 
        
    return d
      
      
      
      
      

def test(filename= 'dataset/GOOG-year.csv'):
    df = pd.read_csv('../dataset/GOOG-year.csv')
    close = df.Close.values.tolist()
    
    ###  Train
    model = Model(window_size, window_size, close, skip, 200, initial_money)
    sess = fit(model, close, do_action_example)
    
    
    ### Predict
    agent.sess = sess
    # states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)
    res_list = predict(model, sess, close, do_action_example)













################################################################################################
################################################################################################
if __name__ == "__main__":


    # In[2]:


    df = pd.read_csv('../dataset/GOOG-year.csv')
    df.head()


    # In[4]:


    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    agent = Agent(state_size = window_size,
                window_size = window_size,
                trend = close,
                skip = skip)
    #agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)


    # In[5]:


    

    # In[6]:
    model = Model(window_size, window_size, close, skip, 200, initial_money)
    sess = fit(model, close, do_action_example)
    agent.sess = sess
    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)
    test()
    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()
    


    # In[ ]:
