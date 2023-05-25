import sys
import os
#dir_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(dir_path)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import scipy.io
import numpy as np
import deepxde as dde # version 0.11 or higher
from deepxde.backend import tf

input = 4 # network input size 
num_hidden_layers = 4 # number of hidden layers for NN 
hidden_layer_size = 32 # size of each hidden layers 
output = 2 # network input size 

num_domain = 40000 # number of training points within the domain
num_boundary = 4000 # number of training boundary condition points on the geometry boundary
num_test = 1000 # number of testing points within the domain
MAX_MODEL_INIT = 16 # maximum number of times allowed to initialize the model
MAX_LOSS = 4 # upper limit to the initialized loss
epochs_init = 15000 # number of epochs for training initial phase
epochs_main = 150000 # number of epochs for main training phase
lr = 0.0005 # learning rate

a = 0.01
b = 0.15
D = 0.1
k = 8
mu_1 = 0.2
mu_2 = 0.3
epsilon = 0.002

## Geometry Parameters

file_name = "10x10x50_corner.mat"
data = scipy.io.loadmat(file_name)

t, x, y, z, Vsav, Wsav = data["t"], data["x"], data["y"], data["z"], data["Vsav"], data["Wsav"]
min_x = x[0][0]
max_x = x[0][-1]
min_y = y[0][0]
max_y = y[0][-1]
min_z = z[0][0]
max_z = z[0][-1]
min_t = t[0][0]
max_t = t[0][-1]
spacing = x[0][1]-x[0][0]


X, T, Y, Z = np.meshgrid(x,t,y,z)
Y = Y.reshape(-1, 1)
Z = Z.reshape(-1, 1)

max_t = np.max(t)
max_x = np.max(x)        
X = X.reshape(-1, 1)
T = T.reshape(-1, 1)
V = Vsav.reshape(-1, 1)
W = Wsav.reshape(-1, 1)


observe_x = np.hstack((X, Y, Z, T))

def IC_func(observe_train, v_train):
        
        T_ic = observe_train[:,-1].reshape(-1,1)
        idx_init = np.where(np.isclose(T_ic,1))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]
        return dde.PointSetBC(observe_init,v_init,component=0)

def boundary_func_3d_cube(x, on_boundary):
    #return on_boundary
    return on_boundary and ~(x[0:3]==[min_x,min_y,min_z]).all() and  ~(x[0:3]==[min_x,min_y, max_z]).all() and \
        ~(x[0:3]==[max_x,min_y,min_z]).all() and  ~(x[0:3]==[max_x,min_y,max_z]).all() and  ~(x[0:3]==[min_x,max_y,min_z]).all() \
        and  ~(x[0:3]==[min_x,max_y,max_z]).all() and  ~(x[0:3]==[max_x,max_y,min_z]).all() and  ~(x[0:3]==[max_x,max_y,max_z]).all()

def BC_func(geomtime):
    bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), boundary_func_3d_cube, component=0)
    return bc

observe_train, observe_test, v_train, v_test, w_train, w_test = train_test_split(observe_x,V,W,test_size=0.8) 

geom = dde.geometry.Cuboid([min_x,min_y, min_z], [max_x,max_y, max_z])
timedomain = dde.geometry.TimeDomain(min_t, max_t)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

## Define Boundary Conditions
bc = BC_func(geomtime)

## Define Initial Conditions
ic = IC_func(observe_train, v_train)
    
## Model observed data
observe_v = dde.PointSetBC(observe_train, v_train, component=0)
input_data = [bc, ic, observe_v]
net = dde.maps.FNN([input] + [hidden_layer_size] * num_hidden_layers + [output], "tanh", "Glorot uniform")

def pde_3D(x, y):
    
    V, W = y[:, 0:1], y[:, 1:2]
    dv_dt = dde.grad.jacobian(y, x, i=0, j=3)
    dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_dzz = dde.grad.hessian(y, x, component=0, i=2, j=2)
    dw_dt = dde.grad.jacobian(y, x, i=1, j=3)

        ## Coupled PDE+ODE Equations
    eq_a = dv_dt -  D*(dv_dxx + dv_dyy + dv_dzz) + k*V*(V-a)*(V-1) +W*V 
    eq_b = dw_dt -  (epsilon + (mu_1*W)/(mu_2+V))*(-W -k*V*(V-b-1))
    return [eq_a, eq_b]

pde = pde_3D

pde_data = dde.data.TimePDE(geomtime, pde, input_data,
                            num_domain = num_domain, 
                            num_boundary=num_boundary, 
                            anchors=observe_train,
                            num_test=num_test) 

model = dde.Model(pde_data, net)

model.compile("adam", lr=lr)

def stable_init(model):
    ## Stabalize initialization process by capping the losses
    losshistory, _ = model.train(epochs=1)
    initial_loss = max(losshistory.loss_train[0])
    num_init = 1
    while initial_loss>MAX_LOSS or np.isnan(initial_loss):
        num_init += 1
        model = dde.Model(pde_data, net)
        model.compile("adam", lr=lr)
        losshistory, _ = model.train(epochs=1)
        initial_loss = max(losshistory.loss_train[0])
        if num_init > MAX_MODEL_INIT:
            raise ValueError('Model initialization phase exceeded the allowed limit')
    return 0

out_path = '3D_large_z/'

def train_3_phase(self, out_path):
    init_weights = [0,0,0,0,1]
        
    ## Initial phase
    model.compile("adam", lr=0.0005, loss_weights=init_weights)
    losshistory, train_state = model.train(epochs=epochs_init, model_save_path = out_path)
    ## Main phase
    model.compile("adam", lr=lr)
    losshistory, train_state = model.train(epochs=epochs_main, model_save_path = out_path)
    ## Final phase
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train(model_save_path = out_path)
    return losshistory, train_state

def train(model, out_path):
        
        ## Stabalize initialization process by capping the losses
        #stable_init(model)
      
     
        losshistory, train_state = train_3_phase(model, out_path)
                   
        return model, losshistory, train_state

model, losshistory, train_state = train(model, out_path)

pred = model.predict(observe_test)
v_pred = pred[:,0:1]
rmse_v = np.sqrt(np.square(v_pred - v_test).mean())

print("V rMSE for test data:", rmse_v)

pred_2 = model.predict(observe_x)
v_pred_model = pred_2[:,0:1]
np.savetxt("v_pred_model_large_z.txt",np.hstack((observe_x,v_pred_model)),header="observe_x, v_pred_model")
