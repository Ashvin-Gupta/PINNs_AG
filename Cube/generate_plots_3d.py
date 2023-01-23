import matplotlib.pyplot as plt
import pylab
import numpy as np
import matplotlib.animation as animation
from PIL import Image
import io
#from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
#from pylab import *

def plot_3D(data_list,dynamics, model, fig_name):
    
    #Set all parameters for the plot
    params = {
       'axes.labelsize': 30,
       'font.size': 13,
       'legend.fontsize': 23,
       'axes.titlesize': 28,
       'xtick.labelsize': 25,
       'ytick.labelsize': 25,
       'figure.figsize': [10, 8],
       'font.family': 'Times New Roman',
       }
    plt.rcParams.update(params)

    plot_3D_cell(data_list, dynamics, model, fig_name[1:])
    #plot_3D_gridxy(data_list,dynamics, model, fig_name[1:])
    #plot_3D_gridxz(data_list,dynamics, model, fig_name[1:])
    #plot_3D_grid(data_list,dynamics, model, fig_name[1:])
    return 0

def plot_3D_cell(data_list, dynamics, model, fig_name):
    
    ## Unpack data
    observe_data, observe_train, v_train, v = data_list[0], data_list[1], data_list[2], data_list[3]
    
    ## Pick a random cell to show
    cell_x = dynamics.max_x*0.75
    cell_y = dynamics.max_y*0.75
    cell_z = np.round(dynamics.max_z*0.75,1)
        
    ## Get data for cell
    idx = [i for i,ix in enumerate(observe_data) if (observe_data[i][0:3]==[cell_x,cell_y,cell_z]).all()]
    observe_geomtime = observe_data[idx]
    v_GT = v[idx]
    v_predict = model.predict(observe_geomtime)[:,0:1]
    t_axis = observe_geomtime[:,3]
    
    ## Get data for points used in training process
    idx_train = [i for i,ix in enumerate(observe_train) if (observe_train[i][0:3]==[cell_x,cell_y,cell_z]).all()]
    v_trained_points = v_train[idx_train]
    t_markers = (observe_train[idx_train])[:,3]
    
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
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(fig_name + "_cell_plot_3D.tiff")
    png1.close()
    return 0

def plot_3D_gridxy(data_list,dynamics, model, fig_name):
    
    grid_size = 60
    rand_t = dynamics.max_t/2
    
    ## Get data
    x = np.linspace(dynamics.min_x,dynamics.max_x, grid_size)
    y = np.linspace(dynamics.min_y,dynamics.max_y, grid_size)
    z = np.round(dynamics.max_z*0.75,1)
    t = np.ones_like(x)*rand_t
    
    X, T, Y, Z = np.meshgrid(x,t,y,z)
    X_data = X.reshape(-1,1)
    Y_data = Y.reshape(-1,1)
    Z_data = Z.reshape(-1,1)
    T_data = T.reshape(-1,1)
    data = np.hstack((X_data, Y_data, Z_data, T_data))
    
    v_pred = model.predict(data)[:,0:1]
    X, Y = np.meshgrid(x,y)
    V = np.zeros((grid_size,grid_size))
    for i in range(grid_size):
        V[i,:] = (v_pred[(i*grid_size):((i+1)*grid_size)]).reshape(-1)
    
    ## create figure
    plt.figure()
    contour = plt.contourf(X,Y,V, cmap=plt.cm.bone)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar(contour)
    cbar.ax.set_ylabel('V')
    
    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(fig_name + "_grid_plot_3Dxy.tiff")
    png1.close()
    return 0   

def plot_3D_gridxz(data_list,dynamics, model, fig_name):
    
    grid_size = 25
    rand_t = dynamics.max_t/2
    
    ## Get data
    x = np.linspace(dynamics.min_x,dynamics.max_x, grid_size)
    y = dynamics.max_y*0.75
    z = np.round(dynamics.max_z*0.75,1)
    t = np.ones_like(x)*rand_t
    
    X, T,Y, Z = np.meshgrid(x,t,y,z)
    X_data = X.reshape(-1,1)
    Y_data = Y.reshape(-1,1)
    Z_data = Z.reshape(-1,1)
    T_data = T.reshape(-1,1)
    data = np.hstack((X_data, Y_data, Z_data, T_data))
    
    v_pred = model.predict(data)[:,0:1]
    X, Z = np.meshgrid(x,z)
    V = np.zeros((grid_size,grid_size))
    for i in range(grid_size):
        V[i,:] = (v_pred[(i*grid_size):((i+1)*grid_size)]).reshape(-1)
    
    ## create figure
    plt.figure()
    contour = plt.contourf(X,Z,V, cmap=plt.cm.bone)
    plt.xlabel('x')
    plt.ylabel('z')
    cbar = plt.colorbar(contour)
    cbar.ax.set_ylabel('V')
    
    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(fig_name + "_grid_plot_3Dxz.tiff")
    png1.close()
    return 0

def plot_3D_grid(data_list,dynamics,model,fig_name):
    
    grid_size = 60
    rand_t = dynamics.max_t/2
    
    ## Get data
    x = np.linspace(dynamics.min_x,dynamics.max_x, grid_size)
    y = np.linspace(dynamics.min_y,dynamics.max_y, grid_size)
    z = np.linspace(dynamics.min_z,dynamics.max_z, grid_size)
    t = np.ones_like(x)*rand_t
    
    X, T,Y, Z = np.meshgrid(x,t,y,z)
    X_data = X.reshape(-1,1)
    Y_data = Y.reshape(-1,1)
    Z_data = Z.reshape(-1,1)
    T_data = T.reshape(-1,1)
    data = np.hstack((X_data, Y_data, Z_data, T_data))
    
    v_pred = model.predict(data)[:,0:1]
    X,Y,Z = np.meshgrid(x,y,z)
    V = np.zeros((grid_size,grid_size,grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            V[i,j,:] = (v_pred[((i+j*grid_size)*grid_size):((i+j*grid_size+1)*grid_size)]).reshape(-1)

    
    # creating figures
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)

    # creating the heatmap
    img = ax.scatter(X, Y, Z, marker='x', s=100, c=V, cmap=plt.cm.bone)
    plt.colorbar(img)
    plt.grid()

    # adding title and labels
    ax.view_init(26,109) 
    cbar.ax.get_yaxis().labelpad = 15;
    cbar.ax.set_ylabel('V');
    ax.set_title("V vs space")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(fig_name + "_grid_plot_3D.tiff")
    png1.close()
    return 0
