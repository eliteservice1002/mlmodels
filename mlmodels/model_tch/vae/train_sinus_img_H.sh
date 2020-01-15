#! /bin/sh
# A Simple Shell Script To Run sinus img training
# mind that images need to in a subfolder under [./adata/sinus]/sinus_img/subfolder/*.png
python ./models/Beta_VAE/main.py --dataset sinus_img --seed 1 --lr 1e-4 --beta1 0.9 --beta 0.999 \
    --objective H --model H --batch_size 64 --z_dim 10 --max_iter 1.5e6 \
    --beta 4 --viz_name sinus_img_H_beta4_z10 --dset_dir ./adata/sinus/
