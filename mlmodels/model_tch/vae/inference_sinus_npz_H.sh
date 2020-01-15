#! /bin/sh
# To inference sinus img training with model H
# mind that images need to in a subfolder under [./adata/sinus]/sinus_img/subfolder/*.png
# and checkpoint must with the same name of --viz_name

python ./models/Beta_VAE/main.py --dataset sinus_npz --objective H --model H --viz_name sinus_npz_H_beta4_z10 --train False --viz_on False --dset_dir ./adata/sinus/
