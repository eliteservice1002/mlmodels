#! /bin/sh

python main.py --dataset sinus --seed 1 --lr 1e-4 --beta1 0.9 --beta 0.999 \
    --objective H --model H --batch_size 64 --z_dim 10 --max_iter 1.5e6 \
    --beta 4 --viz_name sinus_H_beta4_z10   --dset_dir  adata/sinus/100_50/
