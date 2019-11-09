#!/usr/bin/env python
# coding: utf-8

# NBEATS EXAMPLE
# https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line
# It's a toy example to show how to do time series forecasting using N-Beats.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import os




from trainer import * # some import from the trainer script e.g. load/save functions.

# https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line
# It's a toy example to show how to do time series forecasting using N-Beats.


# In[3]:


# plot utils.
def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


# simple batcher.
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






# trainer
def train_100_grad_steps(data, device, net, optimiser, test_losses):
    global_step = load(net, optimiser)
    for x_train_batch, y_train_batch in data:
        global_step += 1
        optimiser.zero_grad()
        net.train()
        _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        if global_step % 30 == 0:
            print(f'grad_step = {str(global_step).zfill(6)}, tr_loss = {loss.item():.6f}, te_loss = {test_losses[-1]:.6f}')
        if global_step > 0 and global_step % 100 == 0:
            with torch.no_grad():
                save(net, optimiser, global_step)
            break


# In[9]:


# evaluate model on test data and produce some plots.
def eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test):
    net.eval()
    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    test_losses.append(F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item())
    p = forecast.detach().numpy()
    subplots = [221, 222, 223, 224]
    plt.figure(1)
    for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
        ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
        plt.subplot(subplots[plot_id])
        plt.grid()
        plot_scatter(range(0, backcast_length), xx, color='b')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
    plt.show()


# In[6]:


# main
#os.remove('nbeats-training-checkpoint.th')


device = torch.device('cpu')  # use the trainer.py to run on GPU.
forecast_length = 5
backcast_length = 3 * forecast_length
batch_size = 10  # greater than 4 for viz





def data_load()
    milk = pd.read_csv('milk.csv', index_col=0, parse_dates=True)

    print(milk.head())
    milk = milk.values  # just keep np array here for simplicity.
    norm_constant = np.max(milk)
    milk = milk / norm_constant  # small leak to the test set here.

    x_train_batch, y = [], []
    for i in range(backcast_length, len(milk) - forecast_length):
        x_train_batch.append(milk[i - backcast_length:i])
        y.append(milk[i:i + forecast_length])

    x_train_batch = np.array(x_train_batch)[..., 0]
    y = np.array(y)[..., 0]

    c = int(len(x_train_batch) * 0.8)
    x_train, y_train = x_train_batch[:c], y[:c]
    x_test, y_test = x_train_batch[c:], y[c:]

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)


# In[6]:


# model
net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                forecast_length=forecast_length,
                thetas_dims=[7, 8],
                nb_blocks_per_stack=3,
                backcast_length=backcast_length,
                hidden_layer_units=128,
                share_weights_in_stack=False,
                device=device)




optimiser = optim.Adam(net.parameters())


# In[7]:


# data
data = data_generator(x_train, y_train, batch_size)


# In[8]:



def fit() :
    # training
    # model seems to converge well around ~2500 grad steps and starts to overfit a bit after.
    test_losses = []
    for i in range(30):
        eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test)
        train_100_grad_steps(data, device, net, optimiser, test_losses)

