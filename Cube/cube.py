
import pyvista as pv
import numpy as np
from itertools import islice

# Open the file and get the total number of lines
with open('v_pred_model.txt', 'r') as f:
    total_lines = sum(1 for line in f)

# Reset the file pointer to the beginning of the file
with open('v_pred_model.txt', 'r') as f:
    # Loop through the file in increments of 60,000 lines
    for i in range(0, total_lines, 60000):
        # Read the next 60,000 lines of data
        data = [[float(num) for num in line.split()] for line in islice(f, 60000)]
        
        # Extract the x, y, z, and U values from the data
        u = [row[4] for row in data]
        val = np.array(u)
        values = val.reshape((100, 100, 6))
        
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
        polydata.save(f"grid_{i/60000}.vtk")

