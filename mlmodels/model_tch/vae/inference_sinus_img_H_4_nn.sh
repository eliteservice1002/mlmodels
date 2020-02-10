#! /bin/sh
# To inference sinus img training with modiefied model H with 4 output
# mind that images need to in a subfolder under [./adata/sinus]/sinus_img/subfolder/*.png
# and checkpoint must with the same name of --viz_name

python ./models/Beta_VAE/main.py --dataset sinus_img --objective H --model H_4_nn --viz_name sinus_img_H_4_nn_beta4_z10 --train False --viz_on False --dset_dir ./adata/sinus/

