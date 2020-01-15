#! /bin/sh
# A Simple Shell Script To Run sinus npz training
# mind that npz file needs to be under [./adata/sinus]/sinus_npz/sinus.npz
python ./models/Beta_VAE/main.py --dataset sinus_npz --seed 1 --lr 1e-4 --beta1 0.9 --beta 0.999 \
    --objective H --model H --batch_size 64 --z_dim 10 --max_iter 1.5e6 \
    --beta 4 --viz_name sinus_npz_H_beta4_z10   --dset_dir  ./adata/sinus/
