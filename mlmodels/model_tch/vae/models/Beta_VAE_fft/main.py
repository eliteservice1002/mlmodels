"""main.py"""

import argparse

import numpy as np
import torch

from solver import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


####################################################################################################
def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)

    if args.train:
        net.train()
    else:
        net.viz_traverse()


####################################################################################################
if __name__ == "__main__":
    def to(*a, **k) :
        p.add_argument(*a, **k)

    p = argparse.ArgumentParser(description='toy Beta-VAE')

    to('--train', default=True, type=str2bool, help='train or traverse')
    to('--seed', default=1, type=int, help='random seed')
    to('--cuda', default=True, type=str2bool, help='enable cuda')
    to('--max_iter', default=1e6, type=float, help='maximum training iteration')
    to('--batch_size', default=64, type=int, help='batch size')

    ## VAE
    to('--z_dim', default=10, type=int, help='dimension of the representation z')
    to('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
    to('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    to('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    to('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
    to('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    to('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
    to('--lr', default=1e-4, type=float, help='learning rate')
    to('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    to('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    #### Data
    to('--dset_dir', default='data', type=str, help='dataset directory')
    to('--dataset', default='CelebA', type=str, help='dataset name')
    to('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    to('--num_workers', default=2, type=int, help='dataloader num_workers')


    to('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    to('--viz_name', default='main', type=str, help='visdom env name')
    to('--viz_port', default=8097, type=str, help='visdom port number')
    to('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    to('--output_dir', default='outputs', type=str, help='output directory')

    to('--gather_step', default=1000, type=int, help='numer of iterations after which data is gathered for visdom')
    to('--display_step', default=10000, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    to('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is saved')

    to('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    to('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')


    args = p.parse_args()

    main(args)
