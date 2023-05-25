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
import matplotlib as mp
import pylab
import warnings
warnings.filterwarnings("ignore")

## PDE Parameters (initialized for 1D PINN)

input = 3
num_hidden_layers = 3
hidden_layer_size = 12
num_domain = 40000
num_boundary = 4000
epochs_main = 150000
output = 2 # network input size 
        
## Training Parameters
num_domain = 20000 # number of training points within the domain
num_boundary = 1000 # number of training boundary condition points on the geometry boundary
num_test = 1000 # number of testing points within the domain
MAX_MODEL_INIT = 16 # maximum number of times allowed to initialize the model
MAX_LOSS = 0.1 # upper limit to the initialized loss
epochs_init = 15000 # number of epochs for training initial phase
epochs_main =400000 # number of epochs for main training phase
lr = 0.0005 # learning rate

a = 0.01
b = 0.15
D = 0.1
k = 8
mu_1 = 0.2
mu_2 = 0.3
epsilon = 0.002

file_name = "spiral400.mat"
data = scipy.io.loadmat(file_name,squeeze_me=True)

t, x, y, Vsav, Wsav = data["t"], data["x"], data["y"], data["Vsav"], data["Wsav"]

## Geometry Parameters
min_x = x[0]
max_x = x[-1]
min_y = y[0]
max_y = y[-1]
min_t = t[0]
max_t = 400
spacing = x[1]-x[0]

X, T, Y = np.meshgrid(x,t,y)
Y = Y.reshape(-1, 1)
max_t = np.max(t)
max_x = np.max(x)        
X = X.reshape(-1, 1)
T = T.reshape(-1, 1)
V = Vsav.reshape(-1, 1)
W = Wsav.reshape(-1, 1)

observe_x = np.hstack((X, Y, T))

def IC_func(observe_train, v_train):
        
        T_ic = observe_train[:,-1].reshape(-1,1)
        idx_init = np.where(np.isclose(T_ic,1))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]
        return dde.PointSetBC(observe_init,v_init,component=0)


def boundary_func_2d(x, on_boundary):
    return on_boundary and ~(x[0:2]==[min_x,min_y]).all() and  ~(x[0:2]==[min_x,max_y]).all() and ~(x[0:2]==[max_x,min_y]).all()  and  ~(x[0:2]==[max_x,max_y]).all() 

def BC_func(geomtime):
        bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), boundary_func_2d, component=0)
        return bc

observe_train, observe_test, v_train, v_test, w_train, w_test = train_test_split(observe_x,V,W,test_size=0.025)  

geom = dde.geometry.Rectangle([min_x,min_y], [max_x,max_y])
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

def pde_2D(x, y):
    
    V, W = y[:, 0:1], y[:, 1:2]
#
    dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
    dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        ## Coupled PDE+ODE Equations
    eq_a = dv_dt -  D*(dv_dxx + dv_dyy) + k*V*(V-a)*(V-1) +W*V 
    eq_b = dw_dt -  (epsilon + (mu_1*W)/(mu_2+V))*(-W -k*V*(V-b-1))
    return [eq_a, eq_b]

pde = pde_2D

pde_data = dde.data.TimePDE(geomtime, pde, input_data,
                            num_domain = num_domain, 
                            num_boundary=num_boundary, 
                            anchors=observe_train,
                            num_test=num_test) 
model = dde.Model(pde_data, net)
model.compile("adam", lr=lr)

def stable_init(model):
    ## Stabilize initialization process by capping the losses
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
out_path = 'spiral_400/'

def train_3_phase(out_path):
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

def train_1_phase(out_path):
    losshistory, train_state = model.train(epochs=epochs_main, model_save_path = out_path)
    return losshistory, train_state

# stable_init(model)
      
# losshistory, train_state = train_3_phase(model, out_path)
losshistory, train_state = train_3_phase(out_path)

pred = model.predict(observe_test)
v_pred = pred[:,0:1]
rmse_v = np.sqrt(np.square(v_pred - v_test).mean())
print("V rMSE for test data:", rmse_v)

pred_2 = model.predict(observe_x)
v_pred_model = pred_2[:,0:1]
np.savetxt("v_pred_model_spiral.txt",np.hstack((observe_x,v_pred_model)),header="observe_x, v_pred_model")



#FUTURE
file_name_future = "Spiral500.mat"
data_f = scipy.io.loadmat(file_name_future,squeeze_me=True)
t_f, x_f, y_f, Vsav_f, Wsav_f = data_f["t"], data_f["x"], data_f["y"], data_f["Vsav"], data_f["Wsav"]
min_x_f = x_f[0]
max_x_f = x_f[-1]
min_y_f = y_f[0]
max_y_f = y_f[-1]
min_t_f = t_f[0]

X_f, T_f, Y_f = np.meshgrid(x_f,t_f,y_f)
Y_f = Y_f.reshape(-1, 1)
max_t_f = np.max(t_f)
max_x_f = np.max(x_f)        
X_f = X_f.reshape(-1, 1)
T_f = T_f.reshape(-1, 1)
V_f = Vsav_f.reshape(-1, 1)
W_f = Wsav_f.reshape(-1, 1)

observe_x_f = np.hstack((X_f, Y_f, T_f))
observe_train_f, observe_test_f, v_train_f, v_test_f, w_train_f, w_test_f = train_test_split(observe_x_f,V_f,W_f,test_size=0.025)
observe_v_f = dde.PointSetBC(observe_train_f, v_train_f, component=0)

pred_f = model.predict(observe_test_f)
v_pred_f = pred_f[:,0:1]
rmse_v_f = np.sqrt(np.square(v_pred_f - v_test_f).mean())
print("V rMSE for test data future:", rmse_v_f)

pred_2_f = model.predict(observe_x_f)
v_pred_model_f = pred_2_f[:,0:1]
np.savetxt("v_pred_model_spiral_future.txt",np.hstack((observe_x_f,v_pred_model_f)),header="observe_x, v_pred_model")





