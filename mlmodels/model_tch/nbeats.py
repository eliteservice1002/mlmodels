import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import argparse
import sys
import matplotlib.pyplot as plt


from torch import nn

import torch
from torch import optim
from torch.nn import functional as F


CHECKPOINT_NAME = 'nbeats-fiting-checkpoint.th'
VERBOSE = True

###############################################################################################################
###############################################################################################################
# Helper functions
####################################################################################################

def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(filepath).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path

def plot(net, x, target, backcast_length, forecast_length, grad_step, out_path="./"):
    net.eval()
    _, f = net(torch.tensor(x, dtype=torch.float))
    subplots = [221, 222, 223, 224]

    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for i in range(4):
        ff, xx, yy = f.cpu().numpy()[i], x[i], target[i]
        plt.subplot(subplots[i])
        plt.plot(range(0, backcast_length), xx, color='b')
        plt.plot(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plt.plot(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        # plt.title(f'step #{grad_step} ({i})')

    output = f'{out_path}/n_beats_{grad_step}.png'
    plt.savefig(output)
    plt.clf()
    print('Saved image to {}.'.format(output))

def log(*s, n=0,m=1):
  sspace = "#" * n
  sjump =  "\n" * m
  print(sjump, sspace, s, sspace, flush=True)

###############################################################################################################
###############################################################################################################
# Get dataset
####################################################################################################


def get_dataset(**kwargs):
    data_path = kwargs['data_path']
    train_split_ratio = kwargs.get("train_split_ratio", 1)

    milk = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if VERBOSE :
      print(milk.head(5))

    milk = milk.values  # just keep np array here for simplicity.
    norm_constant = np.max(milk)
    milk = milk / norm_constant  # small leak to the test set here.



    x_train_batch, y = [], []
    backcast_length=kwargs['backcast_length']
    forecast_length = kwargs['forecast_length']
    for i in range(backcast_length, len(milk) - forecast_length):
        x_train_batch.append(milk[i - backcast_length:i])
        y.append(milk[i:i + forecast_length])

    x_train_batch = np.array(x_train_batch)[..., 0]
    y = np.array(y)[..., 0]

    #### Split
    c = int(len(x_train_batch) * train_split_ratio)
    x_train, y_train = x_train_batch[:c], y[:c]
    x_test, y_test = x_train_batch[c:], y[c:]
    return x_train,y_train,x_test,y_test,norm_constant



def data_generator(x_full, y_full, bs):
    def split(arr, size):
        arrays = []
        while len(arr) > size:
            slice_ = arr[:size]
            arrays.append(slice_)
            arr = arr[size:]
        arrays.append(arr)
        return arrays

    while True:
        for rr in split((x_full, y_full), bs):
            yield rr



###############################################################################################################
###############################################################################################################
# Model
####################################################################################################
class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 device,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 thetas_dims=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256):
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dims
        self.parameters = []
        self.device = device
        print(f'| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id],
                                   self.device, self.backcast_length, self.forecast_length)
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
        return backcast, forecast


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p < 10, 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5):
        super(SeasonalityBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                               forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5):
        super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5):
        super(GenericBlock, self).__init__(units, thetas_dim, device, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast



###############################################################################################################
###############################################################################################################
# Model fit
####################################################################################################


def fit(model, data_pars, compute_pars={}, out_pars=None,  **kwargs):
    device = torch.device('cpu')
    forecast_length = data_pars["forecast_length"]
    backcast_length = data_pars["backcast_length"]

    batch_size = compute_pars["batch_size"]  # greater than 4 for viz
    disable_plot=compute_pars["disable_plot"]
    
    x_train, y_train, x_test, y_test,_ = get_dataset(**data_pars)
    
    data_gen = data_generator(x_train, y_train, batch_size)
    net = model

    optimiser = optim.Adam(net.parameters())

    def plot_model(x, target, grad_step):
        if not disable_plot:
            print('plot()')
            plot(net, x, target, backcast_length, forecast_length, grad_step)

    fit_simple(net, optimiser, data_gen, plot_model, device)


###############################################################################################################
###############################################################################################################
# Partial fit
####################################################################################################
def predict( net,data_pars, compute_pars={}, out_pars=None,  **kwargs):
    forecast_length = data_pars["forecast_length"]
    backcast_length = data_pars["backcast_length"]
    norm_constant = compute_pars["norm_contsant"]
    output = out_pars['output']
    
    data_pars["train_split_ratio"] = 1

    # x_train, y_train, x_test, y_test,_ = get_dataset(**data_pars)
    
    x_test,_,_,_ = get_dataset(**data_pars)

    test_losses=[]
    net.eval()
    _, f = net(torch.tensor(x_test, dtype=torch.float))
    test_losses.append(F.mse_loss(f, torch.tensor(y_test, dtype=torch.float)).item())
    p = f.detach().numpy()
    return p



def plot_predict(p, data_pars, compute_pars, out_pars) :
    forecast_length = data_pars["forecast_length"]
    backcast_length = data_pars["backcast_length"]
    norm_constant = compute_pars["norm_contsant"]
    output = out_pars['output']

    subplots = [221, 222, 223, 224]
    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for plot_id, i in enumerate(np.random.choice(range(len(p)), size=4, replace=False)):
        ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
        plt.subplot(subplots[plot_id])
        plt.grid()
        plt.plot(range(0, backcast_length), xx, color='b')
        plt.plot(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plt.plot(range(backcast_length, backcast_length + forecast_length), ff, color='r')
    plt.savefig(output)
    plt.clf()
    print('Saved image to {}.'.format(output))
    return test_losses




def fit_simple(net, optimiser, data_generator, on_save_callback, device, max_grad_steps=2000):
    print('--- fiting ---')
    initial_grad_step = load(net, optimiser)
    for grad_step, (x, target) in enumerate(data_generator):
        grad_step += initial_grad_step
        optimiser.zero_grad()
        net.train()
        backcast, forecast = net(torch.tensor(x, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        print(f'grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}')
        if grad_step % 100 == 0 or (grad_step < 100 and grad_step % 100 == 0):
            with torch.no_grad():
                save(net, optimiser, grad_step)
                if on_save_callback is not None:
                    on_save_callback(x, target, grad_step)
        if grad_step > max_grad_steps:
            print('Finished.')
            break



###############################################################################################################
###############################################################################################################
# save and load model helper function
####################################################################################################

def save(model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, CHECKPOINT_NAME)


def load(model, optimiser):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0


######################################################################################
def test2(data_path="dataset/milk.csv",out_path="n_beats_test{}.png", reset=True):


    ###loading the command line arguments
   # arg = load_arguments()
    data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
    print(data_path)
    data_pars = {"data_path": data_path, "forecast_length": 5, "backcast_length": 10}

    ##loading dataset
    x_train, y_train, x_test, y_test, norm_const = get_dataset(**data_pars)

    ## loading model

    device = torch.device('cpu')
    model_pars = {"stack_types": [NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK], "device": device,
                  "nb_blocks_per_stack": 3, "forecast_length": 5, "backcast_length": 10,
                  "thetas_dims": [7, 8], "share_weights_in_stack": False, "hidden_layer_units": 256}

    out_pars = {"path": data_path + out_path}
    compute_pars = {}
    #### Model setup, fit, predict
    log("############ Model preparation   #########################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full("nbeats-fiting-checkpoint.th", model_pars)
    print(module, model)

    log("############ Model fit   ##################################")
    sess = fit(model, module, data_pars=data_pars, out_pars=out_pars, compute_pars={})
    print("fit success", sess)

    log("############ Prediction##########################")
    preds = predict(model, module, sess, data_pars=data_pars,
                    out_pars=out_pars, compute_pars=compute_pars)
    print(preds)



from  nbeats.n_beats.model import *


def test(data_path="dataset/milk.csv"):

    ###loading the command line arguments
    data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
    print(data_path)
    data_pars = {"data_path":data_path,"forecast_length": 5, "backcast_length": 10}

    ##loading dataset
    x_train, y_train, x_test, y_test,norm_const = get_dataset(**data_pars)

    ## loading model

    device = torch.device('cpu')
    model_pars =  {"stack_types":[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                   "device":device,
                   "nb_blocks_per_stack":3,"forecast_length": 5, "backcast_length": 10,
                   "thetas_dims":[7, 8],"share_weights_in_stack":False,"hidden_layer_units":256}


    #### Model setup, fit, predict
    model = NBeatsNet(**model_pars)
    compute_pars={"batch_size":100,"disable_plot":False,
                  "norm_contsant":norm_const,
                  "result_path" :'n_beats_test{}.png',
                  "model_path": CHECKPOINT_NAME}
    out_pars = {}
    fit(model, data_pars, compute_pars)

    #### Predict
    ypred = predict(model, data_pars, compute_pars, out_pars)
    plot_predict(ypred)
    print(ypred)





if __name__ == '__main__':
    test()