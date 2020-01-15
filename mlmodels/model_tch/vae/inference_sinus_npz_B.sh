#! /bin/sh
# To inference sinus npz training with model B
# mind that images need to in a subfolder under [./adata/sinus]/sinus_img/subfolder/*.png
# and checkpoint must with the same name of --viz_name

python ./models/Beta_VAE/main.py --dataset sinus_npz --objective B --model B --viz_name sinus_npz_B_gamma100_z10 --train False --viz_on False --dset_dir ./adata/sinus/

