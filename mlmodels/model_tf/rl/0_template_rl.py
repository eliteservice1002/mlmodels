# coding: utf-8
"""
RF framework
https://github.com/google/dopamine/tree/master/docs
https://github.com/deepmind/trfl


""""
import numpy as np
import pandas as pd
import tensorflow as tf


class to_name(object):
  def __init__(self, adict):
    self.__dict__.update(adict)
    
    
    
def val(x,y) :
  return x if x is not None else y
	

class Model:
    def __init__(self, history, params={}):
        p = to_name(params)
        self.agent = Agent(history, params)
        self.n_iter = p.n_iter
        self.initial_reward = p.initial_reward
        ##

class Agent:
    def __init__(self, history, do_action, params={}):
        p = to_name(params)
        self.LEARNING_RATE = p.learning_rate
        self.OUTPUT_SIZE   = p.output_size
        self.GAMMA   = p.GAMMA
        
        self.history = history
 
 
        ### State, reward, action
        self.X       = None   # n_timestep x n_statesize
        self.REWARDS = None 
        self.ACTIONS = None
       
       
        self.logits = None
       
        self.cost = None
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


    def predict_action(self, inputs):
        # Predict probabiulity of all actions
        return self.sess.run(self.logits, feed_dict={self.X: inputs})


    def get_predicted_action(self, sequence):
        prediction = self.predict_action(np.array(sequence))[0]  #Proba vector for each action
        return np.argmax(prediction)
    
    
    def get_state(self, t, state=None, history=None, reward=None):
        """
         Action ---> ENV -->  Reward +Change in State== reward state 
         In this particular case, there is no rewaed state
         State is difference in price over window
        
        """
        state = {}
        return state
        
        
    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.GAMMA + r[t]
            discounted_r[t] = running_add
        return discounted_r
    
    
    def run_sequence(self, history, do_action, params):
        """
          Generate the states, and get action result intoList
        
          history     :  timeseries of FIXED data
          do_action   :  action to be done 
          param       :  parameters into dict

        """
        p = params
        state      = p.get("state_initial")
        reward     = p.get("reward_initial")
        step = p["step"]
        ep_history = []
        
        for t in range(0, len(history) - 1, step):
            action_t    = self.get_predicted_action(t + 1, state, history, reward)
       
            action_dict = {"t": t, 
                           "action":   action_t,             
                           "state":    state,    
                           "history":  history,           
                           "reward" :  reward,
                          }
       
            d            = do_action( action_dict )  
            
            state        = merge_state(state, d)
            reward       = d["reward"]
            
            next_state  = self.get_state(t + 1, state, history, reward)  # can be None
            ep_history.append([state, action_t, reward, next_state])
            state = next_state


        ep_history      = np.array(ep_history)
        ep_history[:,2] = agent.discount_rewards(ep_history[:,2])
        return ep_history
   
   
    def train(self, n_iters=1, n_log_freq=1, state_initial=None, reward_initial=None):
        # Old version
        for i in range(n_iters):
            ep_history = []
            state      = state_initial
            reward     = reward_initial

            ep_history = run_sequence(self.history, do_action , 
                                          {"state_initial" :  self.state_initial, 
                                           "reward_initial" : self.reward_initial, 
                                           "step" : self.step})
                        
            cost, _ = self.sess.run([self.cost, self.optimizer], 
                                     feed_dict={self.X:       np.vstack(ep_history[:,0]),
                                                self.REWARDS: ep_history[:,2],
                                                self.ACTIONS: ep_history[:,1]})
            
            if (i+1) % n_log_freq == 0:
                print('epoch: %d, cost: %f'%(i + 1, cost))

    
 
def fit(model, df, do_action, state_initial=None, reward_initial=None, params=None):
    agent = model.agent(history=df.values, do_action=do_action,
                        params= params)
    agent.train()
    return agent.sess



def predict(model, sess, df, do_action=None,  params= params) :
    model.agent.sess = sess
    res = model.agent.run_sequence(df.values, do_action, params= params) #TODO needs an example function to work
    return res




################################################################################################
################################################################################################
def do_action_example(action_dict):
    """
    
    """
    x         = to_name(action_dict)

    ########## Mapping ####################################
    t            = x.t # time step
    action       = x.action  #selectec action
    Ht           = x.history[x.t]   # current price from history price
    reward       = x.reward
    

    
    #######################################################
    #### Buy
    if action == 1 :
      reward = 1

    ### Sell    
    elif action == 2 :
      reward = 0

    ##### Mapping Back #################################
    invest = ((initial_money - starting_money) / starting_money) * 100
    total_gains = initial_money - starting_money
    
    
    d = { "inventory"      : inventory,
          
          "reward_t"       : invest, 
          "total_reward"   : total_gains,
          "reward_state"   : None, 
          

        } 
        
    return d
      
      
      
      
      










################################################################################################
################################################################################################
if __name__ == "__main__":
    df = pd.read_csv('../dataset/GOOG-year.csv')
    df.head()


    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    agent = Agent(state_size = window_size,
                window_size = window_size,
                trend = close,
                skip = skip)
    #agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)



    model = Model(window_size, window_size, close, skip, 200, initial_money)
    sess = fit(model, close, do_action_example)
    agent.sess = sess
    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)
    test()

    

