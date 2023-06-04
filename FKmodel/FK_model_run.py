import sys
import os
import csv
#dir_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(dir_path)
#dir_path = '/content/drive/MyDrive/Annie'
#sys.path.append(dir_path)
from sklearn.model_selection import train_test_split
from deepxde.backend import tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import deepxde as dde # version 0.11 or higher
#from generate_plot import plot_1D  # should be changed for the new one
#from generate_plots_2d import plot_2D
#from generate_plots_1d_MV import plot_1D
import utils

input_2d = 3 # network input size (2D)
num_hidden_layers_2d = 4 # number of hidden layers for NN (2D)
hidden_layer_size_2d = 32 # size of each hidden layers (2D)
output_2d = 3 # network output size (2D)
output_heter = 3 # network output size for heterogeneity case (2D)

## Training Parameters
num_domain = 20000 # number of training points within the domain
num_boundary = 1000 # number of training boundary condition points on the geometry boundary
num_test = 1000 # number of testing points within the domain
num_initial = 98 # number of points to test the initial conditions
MAX_MODEL_INIT = 16 # maximum number of times allowed to initialize the model
MAX_LOSS = 4 # upper limit to the initialized loss
epochs = 100000 #60000 # number of epochs for training
lr =  0.00005 # learning rate
noise = 0.1 # noise factor
test_size = 0.1 # precentage of testing data 

dim = 2
noise_introduced = False
inverse_activated = True
inverse_string = 'fix_tauo'
inverse = ['taud','taur','tausi']  #[inverse_string]
#model_folder_name = '/content'
animation = False
heter = False
w_used = False #data

dynamics = utils.system_dynamics()

params = dynamics.params_to_inverse(inverse)

file_name =  'U.mat'
observe_x, u = dynamics.generate_data(file_name, dim)

train_mask = np.zeros(len(u), dtype=bool)
nlocs = len(np.unique(observe_x[:,0]))   #number of unique locations with observations
for i in range(nlocs):
    train_mask[i::10] = True

u_test = u[train_mask]
u_train = u[~train_mask]

observe_test = observe_x[train_mask]
observe_train = observe_x[~train_mask]

geomtime = dynamics.geometry_time(dim)
bc = dynamics.BC_func(dim, geomtime)
ic_u, ic_v, ic_w = dynamics.IC_func(observe_train, u_train, geomtime)

observe_u = dde.PointSetBC(observe_train, u_train, component=0)  # component says which component it is
input_data = [bc, ic_u, ic_v, ic_w, observe_u]

pde = dynamics.pde_2D

net = dde.maps.FNN([input_2d] + [hidden_layer_size_2d] * num_hidden_layers_2d + [output_2d], "tanh", "Glorot uniform")
pde_data = dde.data.TimePDE(geomtime, pde, input_data,
                            num_domain = num_domain,
                            num_boundary=num_boundary,
                            num_initial = num_initial,
                            anchors=observe_train,
                            num_test=num_test)
model = dde.Model(pde_data, net)

print("flag 1")
model.compile("adam", lr=lr)
print("flag 2")

out_path = 'train'

losshistory, train_state = model.train(epochs=epochs, model_save_path = out_path, display_every=1000)