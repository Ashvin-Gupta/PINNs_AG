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

import warnings
warnings.filterwarnings("ignore")

## PDE Parameters (initialized for 1D PINN)

input = 3
num_hidden_layers = 5
hidden_layer_size = 60
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
epochs_main =200000 # number of epochs for main training phase
lr = 0.0005 # learning rate

a = 0.01
b = 0.15
D = 0.1
k = 8
mu_1 = 0.2
mu_2 = 0.3
epsilon = 0.002

file_name = "Spiral_Marta.mat"
data = scipy.io.loadmat(file_name,squeeze_me=True)
t, x, y, Vsav, Wsav = data["t"], data["x"], data["y"], data["Vsav"], data["Wsav"]

## Geometry Parameters
min_x = x[0]
max_x = x[-1]
min_y = y[0]
max_y = y[-1]
min_t = y[0]
max_t = t[-1]
spacing = x[1]-x[0]
# print(spacing)

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

observe_train, observe_test, v_train, v_test, w_train, w_test = train_test_split(observe_x,V,W,test_size=0.8)  

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
out_path = 'spiral_2/'

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

def plot_2D(data_list, model, fig_name):

    plot_2D_cell(data_list, model, fig_name[1:])
    plot_2D_grid(data_list, model, fig_name[1:])
    generate_2D_animation( model, fig_name[1:])
    return 0
def plot_2D_cell(data_list, model, fig_name):

    ## Unpack data
    observe_data, observe_train, v_train, v = data_list[0], data_list[1], data_list[2], data_list[3]

    ## Pick a random cell to show
    cell_x = max_x*0.75
    cell_y = max_y*0.75

    ## Get data for cell
    idx = [i for i,ix in enumerate(observe_data) if (observe_data[i][0:2]==[cell_x,cell_y]).all()]
    observe_geomtime = observe_data[idx]
    v_GT = v[idx]
    v_predict = model.predict(observe_geomtime)[:,0:1]
    t_axis = observe_geomtime[:,2]

    ## Get data for points used in training process
    idx_train = [i for i,ix in enumerate(observe_train) if (observe_train[i][0:2]==[cell_x,cell_y]).all()]
    v_trained_points = v_train[idx_train]
    t_markers = (observe_train[idx_train])[:,2]

    ## create figure
    plt.figure()
    plt.plot(t_axis, v_GT, c='b', label='GT')
    plt.plot(t_axis, v_predict, c='r', label='Predicted')
    # If there are any trained data points for the current cell
    if len(t_markers):
        plt.scatter(t_markers, v_trained_points, marker='x', c='black',s=6, label='Observed')
    plt.legend(loc='upper right')
    plt.xlabel('t')
    plt.ylabel('V')

    ## save figure
    #png1 = io.BytesIO()
    #plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    #png2 = Image.open(png1)
    #png2.save(fig_name + "_cell_plot_2D.tiff")
    #png1.close()
    return 0

def plot_2D_grid(data_list, model, fig_name):

    grid_size = 200
    rand_t = max_t/2

    ## Get data
    x = np.linspace(min_x,max_x, grid_size)
    y = np.linspace(min_y,max_y, grid_size)
    t = np.ones_like(x)*rand_t

    X, T, Y = np.meshgrid(x,t,y)
    X_data = X.reshape(-1,1)
    Y_data = Y.reshape(-1,1)
    T_data = T.reshape(-1,1)
    data = np.hstack((X_data, Y_data, T_data))

    v_pred = model.predict(data)[:,0:1]
    X, Y = np.meshgrid(x,y)
    Z = np.zeros((grid_size,grid_size))
    for i in range(grid_size):
        Z[i,:] = (v_pred[(i*grid_size):((i+1)*grid_size)]).reshape(-1)

    ## create figure
    plt.figure()
    contour = plt.contourf(X,Y,Z, cmap=plt.cm.bone)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar(contour)
    cbar.ax.set_ylabel('V')

    ## save figure
    #png1 = io.BytesIO()
    #plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    #png2 = Image.open(png1)
    #png2.save(fig_name + "_grid_plot_2D.tiff")
    #png1.close()
    return 0

def generate_2D_animation( model, fig_name):

    grid_size = 200
    nT = int(max_t)
    n_frames = 2*nT
    x = np.linspace(min_x,max_x, grid_size)
    y = np.linspace(min_y,max_y, grid_size)
    X, Y = np.meshgrid(x,y)
    Z_0 = np.zeros((grid_size,grid_size))

    def get_animation_data(i):
        ## predict V values for each frame (each time step)
        t = np.ones_like(x)*(nT/n_frames)*(i+1)
        X, T, Y = np.meshgrid(x,t,y)
        X_data = X.reshape(-1,1)
        Y_data = Y.reshape(-1,1)
        T_data = T.reshape(-1,1)
        data = np.hstack((X_data, Y_data, T_data))
        v_pred = model.predict(data)[:,0:1]
        Z = np.zeros((grid_size,grid_size))
        for i in range(grid_size):
            Z[i,:] = (v_pred[(i*grid_size):((i+1)*grid_size)]).reshape(-1)
        return Z

    ## Create base screen
    fig = pylab.figure()
    ax = pylab.axes(xlim=(min_x, max_x), ylim=(min_y, max_y), xlabel='x', ylabel='y')
    levels = np.arange(0,1.15,0.1)
    contour = pylab.contourf(X, Y, Z_0, levels = levels, cmap=plt.cm.bone)
    cbar = pylab.colorbar()
    cbar.ax.set_ylabel('V')

    def animate(i):
        ## create a frame
        Z = get_animation_data(i)
        contour = pylab.contourf(X, Y, Z, cmap=plt.cm.bone)
        plt.title('t = %.1f' %((nT/n_frames)*(i+1)))
        return contour

    anim = mp.animation.FuncAnimation(fig, animate, frames=n_frames, repeat=False)
    anim.save(fig_name+'_2D_Animation.mp4', writer=mp.animation.FFMpegWriter(fps=10))
    return 0

data_list = [observe_x, observe_train, v_train, V]
plot_2D(data_list, model, 'test')

