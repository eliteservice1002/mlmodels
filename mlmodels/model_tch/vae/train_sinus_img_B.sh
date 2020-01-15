#! /bin/sh
# A Simple Shell Script To Run sinus img training
# mind that images need to in a subfolder under [./adata/sinus]/sinus_img/subfolder/*.png
python ./models/Beta_VAE/main.py --dataset sinus_img --seed 2 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --objective B --model B --batch_size 64 --z_dim 10 --max_iter 1.5e6 \
    --C_stop_iter 1e5 --C_max 20 --gamma 100 --viz_name sinus_img_B_gamma100_z10 --save_output True --dset_dir ./adata/sinus/
