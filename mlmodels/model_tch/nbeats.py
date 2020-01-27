import os
import pandas as pd
import numpy as np

import torch
from torch import optim
from torch.nn import functional as F


####################################################################################################
CHECKPOINT_NAME = 'nbeats-fiting-checkpoint.th'
VERBOSE = True
from mlmodels.model_tch.nbeats.model import NBeatsNet





####################################################################################################
# Helper functions
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


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)



####################################################################################################
# Model
Model = NBeatsNet




####################################################################################################
# Dataaset
def get_dataset(**kwargs):
    data_path = kwargs['data_path']
    train_split_ratio = kwargs.get("train_split_ratio", 1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if VERBOSE: print(df.head(5))

    #### Preprocess
    df = df.values  # just keep np array here for simplicity.
    norm_constant = np.max(df)
    df = df / norm_constant  # small leak to the test set here.

    x_train_batch, y = [], []
    backcast_length = kwargs['backcast_length']
    forecast_length = kwargs['forecast_length']
    for i in range(backcast_length, len(df) - forecast_length):
        x_train_batch.append(df[i - backcast_length:i])
        y.append(df[i:i + forecast_length])

    x_train_batch = np.array(x_train_batch)[..., 0]
    y = np.array(y)[..., 0]

    #### Split
    c = int(len(x_train_batch) * train_split_ratio)
    x_train, y_train = x_train_batch[:c], y[:c]
    x_test, y_test = x_train_batch[c:], y[c:]
    return x_train, y_train, x_test, y_test, norm_constant


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


######################################################################################################
# Model fit
def fit(model, data_pars, compute_pars={}, out_pars=None, **kwargs):
    device = torch.device('cpu')
    forecast_length = data_pars["forecast_length"]
    backcast_length = data_pars["backcast_length"]
    batch_size = compute_pars["batch_size"]  # greater than 4 for viz
    disable_plot = compute_pars["disable_plot"]

    ### Get Data
    x_train, y_train, x_test, y_test, _ = get_dataset(**data_pars)
    data_gen = data_generator(x_train, y_train, batch_size)

    ### Setup session
    optimiser = optim.Adam(model.parameters())

    ### fit model
    net, optimiser= fit_simple(model, optimiser, data_gen, plot_model, device, data_pars)
    return net, optimiser


def fit_simple(net, optimiser, data_generator, on_save_callback, device, data_pars, max_grad_steps=500):
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
                    on_save_callback(net, x, target, grad_step, data_pars)

        if grad_step > max_grad_steps:
            print('Finished.')
            break
    return net, optimiser

def predict(model, data_pars, compute_pars={}, out_pars=None, **kwargs):
    data_pars["train_split_ratio"] = 1

    x_test, y_test, _, _, _ = get_dataset(**data_pars)

    test_losses = []
    model.eval()
    _, f = model(torch.tensor(x_test, dtype=torch.float))
    test_losses.append(F.mse_loss(f, torch.tensor(y_test, dtype=torch.float)).item())
    p = f.detach().numpy()
    return p


###############################################################################################################
def plot(net, x, target, backcast_length, forecast_length, grad_step, out_path="./"):
    import matplotlib.pyplot as plt
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


def plot_model(net, x, target, grad_step, data_pars, disable_plot=False):
    forecast_length = data_pars["forecast_length"]
    backcast_length = data_pars["backcast_length"]

    # batch_size = compute_pars["batch_size"]  # greater than 4 for viz
    # disable_plot = compute_pars.get("disable_plot", False)

    if not disable_plot:
        print('plot()')
        plot(net, x, target, backcast_length, forecast_length, grad_step)


def plot_predict(x_test, y_test, p, data_pars, compute_pars, out_pars):
    import matplotlib.pyplot as plt
    forecast_length = data_pars["forecast_length"]
    backcast_length = data_pars["backcast_length"]
    norm_constant = compute_pars["norm_contsant"]
    out_path = out_pars['out_path']
    output = f'{out_path}/n_beats_test.png'

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


###############################################################################################################
# save and load model helper function
def save(model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, CHECKPOINT_NAME)


def load(model, optimiser, CHECKPOINT_NAME = 'nbeats-fiting-checkpoint.th'):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0


#############################################################################################################
def test2(data_path="dataset/milk.csv", out_path="n_beats_test{}.png", reset=True):
    ###loading the command line arguments
    # arg = load_arguments()
    data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
    print(data_path)
    data_pars = {"data_path": data_path, "forecast_length": 5, "backcast_length": 10}

    ##loading dataset
    x_train, y_train, x_test, y_test, norm_const = get_dataset(**data_pars)

    ##Params
    device = torch.device('cpu')
    model_pars = {"stack_types": [NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK], "device": device,
                  "nb_blocks_per_stack": 3, "forecast_length": 5, "backcast_length": 10,
                  "thetas_dims": [7, 8], "share_weights_in_stack": False, "hidden_layer_units": 256}

    out_pars = {"path": data_path + out_path}
    compute_pars = {}

    log("############ Model preparation   #########################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full("nbeats-fiting-checkpoint.th", model_pars)
    print(module, model)

    log("############ Model fit   ##################################")
    sess = fit(model, module, data_pars=data_pars, out_pars=out_pars, compute_pars={})
    print("fit success", sess)

    log("############ Prediction  ##################################")
    preds = predict(model, module, sess, data_pars=data_pars,
                    out_pars=out_pars, compute_pars=compute_pars)
    print(preds)


def test(data_path="dataset/milk.csv"):
    ###loading the command line arguments
    data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
    print(data_path)

    ## Loading dataset
    data_pars = {"data_path": data_path, "forecast_length": 5, "backcast_length": 10, "train_split_ratio": 0.8}
    x_train, y_train, x_test, y_test, norm_const = get_dataset(**data_pars)


    ## Model setup
    device = torch.device('cpu')
    model_pars = {"stack_types": [NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                  "device": device,
                  "nb_blocks_per_stack": 3, "forecast_length": 5, "backcast_length": 10,
                  "thetas_dims": [7, 8], "share_weights_in_stack": False, "hidden_layer_units": 256}
    model = NBeatsNet(**model_pars)

    #### Model fit
    compute_pars = {"batch_size": 100, "disable_plot": False,
                    "norm_contsant": norm_const,
                    "result_path": 'n_beats_test{}.png',
                    "model_path": CHECKPOINT_NAME}
    out_pars = {"out_path": "./"}
    fit(model, data_pars, compute_pars)


    #### Predict
    ypred = predict(model, data_pars, compute_pars, out_pars)
    print(ypred)

    #### Plot
    plot_predict(ypred, data_pars, compute_pars, out_pars)




if __name__ == '__main__':
    test()




