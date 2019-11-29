# -*- coding: utf-8 -*-
"""

# Trains an MNIST digit recognizer using PyTorch, and uses tensorboardX to log training metrics
# and weights in TensorBoard event format to the MLflow run's artifact directory. This stores the
# TensorBoard events in MLflow for later access using the TensorBoard command line tool.
#
# NOTE: This example requires you to first install PyTorch (using the instructions at pytorch.org)
#       and tensorboardX (using pip install tensorboardX).
#
# Code based on https://github.com/lanpa/tensorboard-pytorch-examples/blob/master/mnist/main.py.
#


"""

from __future__ import print_function
import argparse
import os


import mlflow
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter


####################################################################################################
from model_tch import *






####################################################################################################
def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)

def tfboard_writer_create() :
    # Create a SummaryWriter to write TensorBoard events locally
    output_dir = dirpath = tempfile.mkdtemp()
    writer = SummaryWriter(output_dir)
    print("Writing TensorBoard events locally to %s\n" % output_dir)
    return writer


def mlflow_add(args) :
    # Log our parameters into mlflow
    for key, value in vars(args).items():
        mlflow.log_param(key, value)



def tfboard_add_weights(step):
        writer.add_histogram('weights/conv1/weight', model.conv1.weight.data, step)
        writer.add_histogram('weights/conv1/bias', model.conv1.bias.data, step)
        writer.add_histogram('weights/conv2/weight', model.conv2.weight.data, step)
        writer.add_histogram('weights/conv2/bias', model.conv2.bias.data, step)
        writer.add_histogram('weights/fc1/weight', model.fc1.weight.data, step)
        writer.add_histogram('weights/fc1/bias', model.fc1.bias.data, step)
        writer.add_histogram('weights/fc2/weight', model.fc2.weight.data, step)
        writer.add_histogram('weights/fc2/bias', model.fc2.bias.data, step)




def session_init(args):
    enable_cuda_flag = True if args.enable_cuda == 'True' else False
    args.cuda = enable_cuda_flag and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  if args.cuda else None
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}




####################################################################################################
def cli_load_arguments():
    # Command-line arguments
    def to(*arg, **kwarg) :
        p.add_argument(*arg, **kwarg)

    p = argparse.ArgumentParser(description='PyTorch MNIST Example')

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
    model = Model()
    model = model.cuda()  if args.cuda else model
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    writer = None # Will be used to write TensorBoard events
    args = cli_load_arguments()
    
    
    enable_cuda_flag = True if args.enable_cuda == 'True' else False
    args.cuda = enable_cuda_flag and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  if args.cuda else None
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    

    with mlflow.start_run():
        # Log our parameters into mlflow
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
    
    
        # Create a SummaryWriter to write TensorBoard events locally
        output_dir = dirpath = tempfile.mkdtemp()
        writer = SummaryWriter(output_dir)
        print("Writing TensorBoard events locally to %s\n" % output_dir)
    

        # Perform the training
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)


        # Upload the TensorBoard event logs as a run artifact
        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(output_dir, artifact_path="events")
        print("\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s" %
            os.path.join(mlflow.get_artifact_uri(), "events"))
    










