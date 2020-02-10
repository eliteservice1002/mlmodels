from util import *
import time

folder = "./adata/sinus/"

start = time.time()
# generate .png with matplotlib
generate_train_img(folder, N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2 ,cmax=2 , step = 0.1, wfreq=0.5 )
elapsed_img = time.time() - start

start = time.time()
# generate .npz with matplotlib
generate_train_npz(folder, N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2 ,cmax=2 , step = 0.1, wfreq=0.5 )
elapsed_npz = time.time() - start

start = time.time()
# generate .png with opencv
generate_train_img_cv(folder, N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2 ,cmax=2 , step = 0.1, wfreq=0.5 )
elapsed_cv_img = time.time() - start

start = time.time()
# generate .npz with opencv

generate_train_npz_cv(folder, N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2 ,cmax=2 , step = 0.1, wfreq=0.5 )
elapsed_cv_npz = time.time() - start

print("img generation with plt takes "+str(elapsed_img)+"s")
print("npz generation with plt takes "+str(elapsed_npz)+"s")
print("img generation with opencv takes "+str(elapsed_cv_img)+"s")
print("npz generation with opencv takes "+str(elapsed_cv_npz)+"s")

print("loaded")


