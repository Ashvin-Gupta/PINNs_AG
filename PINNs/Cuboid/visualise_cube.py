# Script to visulaise the output of EP-PINNs in 3D
# This file opens v_pred_model text file
# Change x, y, z values as appropriate 

import pyvista as pv
import numpy as np
from itertools import islice

x = 180
y = 180
z = 2
loop = x*y*z

# Open the file and get the total number of lines
with open('v_pred_model_3Dchaotic.txt', 'r') as f:
    total_lines = sum(1 for line in f) -1

# Reset the file pointer to the beginning of the file
with open('v_pred_model_3Dchaotic.txt', 'r') as f:
    next(f)
    # Loop through the file in increments of 60,000 lines
    for i in range(0, total_lines, loop):
        # Read the next x*y*z lines of data
        data = [[float(num) for num in line.split()] for line in islice(f, loop)]
        
        # Extract the x, y, z, and U values from the data
        # If comparing the matlab input data, change the 4 to a 1
        u = [row[4] for row in data]
        val = np.array(u)
        values = val.reshape((x, y, z))
        
        # Create the spatial reference
        grid = pv.UniformGrid()
        
        # Set the grid dimensions: shape + 1 because we want to inject our values on
        #   the CELL data
        grid.dimensions = np.array(values.shape) + 1
        
        # Edit the spatial reference
        grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
        grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis
        
        # Add the data values to the cell data
        grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array!
        
        # Convert the grid to a PolyData or StructuredGrid object
        # and save to a vtk file
        polydata = grid.cast_to_unstructured_grid().triangulate()
        polydata.save(f"grid_{i}.vtk")

