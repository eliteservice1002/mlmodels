
import os, sys
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import pandas as pd
import cv2

"""
functionality: sine wave npz generation and image gerneration
"""

# default image shape 64x64x3
# default npz element size ([sample number], 64, 64) 
data = {'resolution': 64, 'amplitude': 5, 'img_relative_folder':"/sinus_img/verbose", \
  'npz_relative_folder':"/sinus_npz", 'npz_name': "sinus.npz", \
  'img_cv_folder':"/sinus_img_cv/verbose", 'npz_cv_folder':"/sinus_npz_cv", 'npz_cv_name': "sinus_cv.npz"}

def set_resolution(resolution = 64):
  data['resolution'] = resolution

def get_resolution():
  return data['resolution']

# sinus: y = a * sin(w * t)
def generate_random_sin(n_rand_starts = 100, amplitude = 1, n_pis = 4, omega = 1, step = 0.2):
  r = np.random.randint(n_rand_starts)
  x = np.arange(r, r + n_pis*np.pi, step)
  y = amplitude * np.sin(omega * x)
  return x,y

# cosinus: y = a * cos (w * t + b) + c
def generate_random_cos(n_rand_starts = 1, a = 1, w= 1, b = 0, c = 0, x_upbound =1, x_downbound = -1, step = 0.2):
  r = np.random.randint(n_rand_starts)
  x = np.arange(x_downbound*2*np.pi+r, x_upbound*2*np.pi+r, step)
  y = a * np.cos(w * x + b) + c

  return x,y

# opencv: create wave image as numpy array
def create_sin_2d_array_cv(x, y, resoltuion = data['resolution'],amp=data['amplitude']):
  size = len(x), len(y), 3
  linewidth = int(len(y)/resoltuion + 0.5) 
  vis = np.zeros(size, dtype=np.uint8)
  new_y = y.copy()
  # amplitude set here for plot
  y_max = amp
  y_min = -1*amp
  border = 16
  ratio = float((len(y)-border) /( y_max - y_min))
  for i in range(len(y)):
    new_y[i] = int(border/2+(len(y)-border)-1-(y[i]-y_min)*ratio)

  pointList = []
  for i in range(int(len(x))):
    pointList.append((i,int(new_y[i])))
  pointList = np.array(pointList)

  cv2.polylines(vis,  [pointList],  False,  (255,255,255),  linewidth)

  vis = cv2.resize(vis, (resoltuion, resoltuion), interpolation=cv2.INTER_CUBIC)
  # threshold as 50
  result = np.where(vis[:,:,0] > 50, 1, 0)
  return result

# opencv: create wave image save as images to disk
def plot_save_disk_cv(x, y, filename, xmax=data['resolution'], ymax=data['resolution'],amp=data['amplitude']):
  size = len(x), len(y), 3
  linewidth = int(len(y)/ymax + 0.5) 
  vis = np.ones(size, dtype=np.uint8)
  vis = vis * 255
  new_y = y.copy()
  y_max = amp
  y_min = -1*amp
  border = 16
  ratio = float((len(y)-border) /( y_max - y_min))
  for i in range(len(y)):
    new_y[i] = int(border/2+(len(y)-border)-1-(y[i]-y_min)*ratio)
  pointList = []
  for i in range(int(len(x))):
    pointList.append((i,int(new_y[i])))
  pointList = np.array(pointList)
  cv2.polylines(vis,  [pointList],  False,  (0,0,0),  linewidth)
  vis = cv2.resize(vis, (xmax, ymax), interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(filename, vis)


# matplotlit: create wave image as numpy array
def create_sin_2d_array_plt(x, y, xmax=data['resolution'], ymax=data['resolution'],amp=data['amplitude']):
  plt.rcParams['axes.facecolor']='white'
  plt.rcParams['savefig.facecolor']='white'
  fig = plt.figure(frameon=False, figsize=(xmax, ymax), dpi=1)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  # amplitude set here for plot
  ax.set_ylim([-1*amp,1*amp])
  ax.set_axis_off()
  fig.add_axes(ax)

  plt.plot(x,y, c="black", linewidth=100)
  fig.canvas.draw()
  frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.clf()
  plt.close('all')
  # frame is binary
  result = np.where(frame[:,:,0] > 254, 0, 1)
  return result

# matplotlib: create wave image save as images to disk
def plot_save_disk(x, y, filename, xmax=data['resolution'], ymax=data['resolution'], amp=data['amplitude']):
  fig = plt.figure(frameon=False, figsize=(xmax, ymax), dpi=1)

  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_ylim([-1*amp,1*amp])
  ax.set_axis_off()
  fig.add_axes(ax)

  plt.plot(x,y, c="black", linewidth=100)
  fig.savefig(filename)
  plt.clf()
  plt.close('all')

# matplotlib: images saves to /path/sinus_img/verbose/*.png
def generate_train_img(folder, N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2 ,cmax=2 , step = 0.1, wfreq=0.5 ) :
    folder_w_subfolder = folder + data['img_relative_folder']
    os.makedirs(folder, exist_ok=True)
    
    os.makedirs(folder_w_subfolder, exist_ok=True)
    folder = os.path.abspath(folder_w_subfolder)

    for type_i in range(N_type):
      for amp_int in range(1, amax*2+1):
        amp_i = amp_int*0.5
        for omega_i in range(wmin, wmax, 1):
          omega_ii = wfreq * omega_i
          for b_i in range(bmin, bmax, 1):
            for c_i in range(cmin, cmax, 1):
              # use sinus gernerate: 
              # x,y = generate_random_sin(N_type, amp_i, 4, omega_ii, step)
              # use cosinus gernerate: 
              x,y = generate_random_cos(n_rand_starts=N_type,a=amp_i, w=omega_ii, b = b_i, c = c_i, step = step)
              filename = '{folder}/sin_{amp_i}-{omega_ii}-{b_i}-{c_i}-{type_i}'.format(folder=folder, amp_i=amp_i, omega_ii=omega_ii, b_i=b_i, c_i=c_i,type_i=type_i).replace(".","_")
              filename = filename + ".png"
              plot_save_disk(x,y, filename, xmax = data['resolution'], ymax = data['resolution'], amp = data['amplitude'])

# matplotlib: images saves to /path/sinus_npz/verbose/sinus.npz
def generate_train_npz(folder, N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2 ,cmax=2 , step = 0.1, wfreq=0.5, resoltuion = data['resolution'] ) :
    folder_w_subfolder = folder + data['npz_relative_folder']
    # inital with empty numpy which is random
    generate_npy = [np.empty([resoltuion, resoltuion], dtype=int)]
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder_w_subfolder, exist_ok=True)
    folder = os.path.abspath(folder)
    is_inital = False

    for type_i in range(N_type):
      for amp_int in range(1, amax*2+1):
        amp_i = amp_int*0.5
        for omega_i in range(wmin, wmax, 1):
          omega_ii = wfreq * omega_i
          for b_i in range(bmin, bmax, 1):
            for c_i in range(cmin, cmax, 1): 
              # use sinus gernerate: 
              # x,y = generate_random_sin(N_type, amp_i, 4, omega_ii, step)
              # use cosinus gernerate: 
              x,y = generate_random_cos(n_rand_starts=N_type,a=amp_i, w=omega_ii, b = b_i, c = c_i, step = step)
              if len(generate_npy) == 1 and is_inital == False:
                # replace the random array with first data, only do once
                generate_npy = [create_sin_2d_array_plt(x,y)]
                is_inital = True
              else:
                generate_npy=np.append(generate_npy, [create_sin_2d_array_plt(x,y)],axis=0)
    np.savez(folder_w_subfolder+ "/"+ data['npz_name'], sinus=generate_npy)
    print("npz saved")

# opencv_version: images saves to /path/sinus_img_cv/verbose/*.png
def generate_train_npz_cv(folder, N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2 ,cmax=2 , step = 0.1, wfreq=0.5, resoltuion = data['resolution'] ) :
    folder_w_subfolder = folder + data['npz_cv_folder']
    # inital with empty numpy which is random
    generate_npy = [np.empty([resoltuion, resoltuion], dtype=int)]
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder_w_subfolder, exist_ok=True)
    folder = os.path.abspath(folder)
    is_inital = False

    for type_i in range(N_type):
      for amp_int in range(1, amax*2+1):
        amp_i = amp_int*0.5
        for omega_i in range(wmin, wmax, 1):
          omega_ii = wfreq * omega_i
          for b_i in range(bmin, bmax, 1):
            for c_i in range(cmin, cmax, 1):
              # use sinus gernerate:                
              # x,y = generate_random_sin(N_type, amp_i, 4, omega_ii, step)
              # use cosinus gernerate: 
              x,y = generate_random_cos(n_rand_starts=N_type,a=amp_i, w=omega_ii, b = b_i, c = c_i, step = step)

              if len(generate_npy) == 1 and is_inital == False:
                # replace the random array with first data, only do once
                generate_npy = [create_sin_2d_array_cv(x,y)]
                is_inital = True
              else:
                generate_npy=np.append(generate_npy, [create_sin_2d_array_cv(x,y)],axis=0)
    np.savez(folder_w_subfolder+ "/"+ data['npz_cv_name'], sinus=generate_npy)
    print("npz saved")

# opencv_version: images saves to /path/sinus_npz/verbose/sinus.npz
def generate_train_img_cv(folder, N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2 ,cmax=2 , step = 0.1, wfreq=0.5) :
    folder_w_subfolder = folder + data['img_cv_folder']
    os.makedirs(folder, exist_ok=True)
    
    os.makedirs(folder_w_subfolder, exist_ok=True)
    folder = os.path.abspath(folder_w_subfolder)

    for type_i in range(N_type):
      for amp_int in range(1, amax*2+1):
        amp_i = amp_int*0.5
        for omega_i in range(wmin, wmax, 1):
          omega_ii = wfreq * omega_i
          for b_i in range(bmin, bmax, 1):
            for c_i in range(cmin, cmax, 1):                
              # use sinus gernerate:                
              # x,y = generate_random_sin(N_type, amp_i, 4, omega_ii, step)
              # use cosinus gernerate: 
              x,y = generate_random_cos(n_rand_starts=N_type,a=amp_i, w=omega_ii, b = b_i, c = c_i, step = step)

              filename = '{folder}/sin_{amp_i}-{omega_ii}-{b_i}-{c_i}-{type_i}'.format(folder=folder, amp_i=amp_i, omega_ii=omega_ii, b_i=b_i, c_i=c_i,type_i=type_i).replace(".","_")
              filename = filename + ".png"
              plot_save_disk_cv(x,y, filename, xmax = data['resolution'], ymax = data['resolution'])

print("loaded")

