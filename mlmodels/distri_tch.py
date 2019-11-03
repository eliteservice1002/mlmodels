"""
Pytorch distributed
### Distributed with OPENMPI with 2 nodes,  model_tch.mlp
./distri_run.sh  2    model_tch.mlp


python   distri_model_tch.py   --model model_tch.mlp


"""
from __future__ import print_function

import argparse
import os

import toml

import horovod.torch as hvd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms


from util import load_config, val
from data import import_data_tch as import_data
from models  import create_instance_tch as create_instance
# from models  import create


#####################################################################################
def load_arguments():
    """
     Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    import argparse

    cur_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(cur_path, "config.toml")

    # Training settings
    p = argparse.ArgumentParser(description="PyTorch MNIST Example")
    p.add_argument("--config_file", default=config_file, help="Params File")
    p.add_argument("--config_mode", default="test", help=" test/ prod /uat")

    p.add_argument("--model", default="model_tch.mlp.py", help=" net")
    p.add_argument("--data", default="mnist", help=" mnist")

    p.add_argument("--batch-size", type=int, default=64, metavar="N", help="batchsize training")
    p.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="batchsize test")
    p.add_argument("--epochs", type=int, default=10, metavar="N", help="num epochs ")
    p.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate ")
    p.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum ")
    p.add_argument("--no-cuda", action="store_true", default=True, help="disables CUDA ")
    p.add_argument("--seed", type=int, default=42, metavar="S", help="random seed ")
    p.add_argument("--log-interval", type=int, default=10, metavar="N", help="log intervl")
    p.add_argument("--fp16-allreduce", action="store_true", default=False, help="fp16 in allreduce")

    ### Should be store in toml file
    p.add_argument("--model_params_name",  default='test', help="model dict_params as Dict")


    args = p.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args = load_config(args, args.config_file, args.config_mode)
    return args


#####################################################################################
# Horovod: initialize library.
args = load_arguments()
hvd.init()
torch.manual_seed(args.seed)
kwargs = {}
if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)
    kwargs = {"num_workers": 1, "pin_memory": True}


#####################################################################################
########### User Specific ###########################################################
train_dataset = import_data(name=args.data, mode="train", node_id=hvd.rank())
test_dataset =  import_data(name=args.data, mode="test", node_id=hvd.rank())


params_dict = args.get( args.get("model_params_name")  )
params_dict = params_dict if params_dict is not None else {} 
model = create_instance(args.model, params=params_dict)  # Net()




#####################################################################################
#####################################################################################
# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
)


# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank()
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs
)


###################################################################################
###################################################################################
if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(), momentum=args.momentum)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(), compression=compression
)


def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print(
                "Local Rank({}) => Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    hvd.local_rank(),
                    epoch,
                    batch_idx * len(data),
                    len(train_sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, "avg_loss")
    test_accuracy = metric_average(test_accuracy, "avg_accuracy")

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
                test_loss, 100.0 * test_accuracy
            )
        )


###################################################################################
###################################################################################
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
