# -*- coding: utf-8 -*-
"""
# Trains an MNIST digit recognizer using PyTorch,
# NOTE: This example requires you to first install PyTorch (using the instructions at pytorch.org)
#       and tensorboardX (using pip install tensorboardX).
#
# Code based on https://github.com/lanpa/tensorboard-pytorch-examples/blob/master/mnist/main.py.




"""

from __future__ import print_function
import argparse
import os
import tempfile


import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from tensorboardX import SummaryWriter
####################################################################################################







########Model definiton ############################################################################
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)

    def log_weights(self, step):
        writer.add_histogram('weights/conv1/weight', model.conv1.weight.data, step)
        writer.add_histogram('weights/conv1/bias', model.conv1.bias.data, step)
        writer.add_histogram('weights/conv2/weight', model.conv2.weight.data, step)
        writer.add_histogram('weights/conv2/bias', model.conv2.bias.data, step)
        writer.add_histogram('weights/fc1/weight', model.fc1.weight.data, step)
        writer.add_histogram('weights/fc1/bias', model.fc1.bias.data, step)
        writer.add_histogram('weights/fc2/weight', model.fc2.weight.data, step)
        writer.add_histogram('weights/fc2/bias', model.fc2.bias.data, step)


######## Generic methods ###########################################################################
def get_pars(choice="test", **kwargs):
    # output parms sample
    # print(kwargs)
    if choice=="test":
        p=         { "learning_rate": 0.001,
            "num_layers": 1,
            "size": None,
            "size_layer": 128,
            "output_size": None,
            "timestep": 4,
            "epoch": 2,
        }

        ### Overwrite by manual input
        for k,x in kwargs.items() :
            p[k] = x

        return p



def get_dataset(data_params) :
    #### Get dataset

    train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),  batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def fit(model, data_pars, compute_pars={}, out_pars=None,  **kwargs):
    train_loader, test_loader = get_dataset(data_params)

    args = to_namepace(compute_params)
    enable_cuda_flag = True if args.enable_cuda == 'True' else False
    args.cuda = enable_cuda_flag and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    model = model.cuda()  if args.cuda else model
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # writer = None # Will be used to write TensorBoard events


    model = train(model, epoch, train_loader)

    # Perform the training
    for epoch in range(1, args.epochs + 1):
        model = train(epoch)
        test(epoch)


    return model, None



def predict(model,sess=None, compute_params=None, data_params=None) :
    # prediction
    train_loader, test_loader = get_dataset(data_params)
    
    return res



def predict_proba(model,  sess=None, compute_params=None, data_params=None) :
    # compute probability, ie softmax
    train_loader, test_loader = get_dataset(data_params)
    
    return res



def metrics(model, sess=None, data_params={}, compute_params={}) :
    # compute some metrics
    pass




def test(arg) :
    # some test runs

    # Create a SummaryWriter to write TensorBoard events locally
    writer = tfboard_writer_create()

    #### Add MLFlow args
    mlflow_add(args)

    check_data = get_dataset()

    model = Model()
    fit(model,module=None, sess=None, compute_params=None, data_params=None)


    # Upload the TensorBoard event logs as a run artifact
    print("Uploading TensorBoard events as a run artifact...")
    mlflow.log_artifacts(output_dir, artifact_path="events")

    print("\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s" %
        os.path.join(mlflow.get_artifact_uri(), "events"))




############################################################################################################
########### Local code, helper code  #######################################################################
def _train(epoch, model, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print( f'Train Epoch: {epoch} [{batch_idx}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * c, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            step = epoch * len(train_loader) + batch_idx
            log_scalar('train_loss', loss.data.item(), step)
            model.log_weights(step)
    return model



def _eval_metrics(model, epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    step = (epoch + 1) * len(train_loader)
    log_scalar('test_loss', test_loss, step)
    log_scalar('test_accuracy', test_accuracy, step)



def _log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)



#######################################################################################
########### CLI #######################################################################
def cli_load_arguments():
    def to(*arg, **kwarg) :
        p.add_argument(*arg, **kwarg)
    p = argparse.ArgumentParser(description='PyTorch CNN')
    to('--do', type=str, default="test",   help='input batch size for training (default: 64)')


    to('--batch-size', type=int, default=64, metavar='N',  help='input batch size for training (default: 64)')
    to('--test-batch-size', type=int, default=1000, metavar='N',  help='input batch size for testing (default: 1000)')
    to('--epochs', type=int, default=10, metavar='N',  help='number of epochs to train (default: 10)')
    to('--lr', type=float, default=0.01, metavar='LR',  help='learning rate (default: 0.01)')
    to('--momentum', type=float, default=0.5, metavar='M',  help='SGD momentum (default: 0.5)')
    to('--enable-cuda', type=str, choices=['True', 'False'], default='True',  help='enables or disables CUDA training')
    to('--seed', type=int, default=1, metavar='S',    help='random seed (default: 1)')
    to('--log-interval', type=int, default=100, metavar='N',    help='how many batches to wait before logging training status')
    args = p.parse_args()
    return args




####################################################################################################
if __name__ == "main" :
    arg = load_arguments()

    if arg.do =="test" :
        test()










