# Open the input files
with open('sphere_test.txt', 'r') as sphere_file, open('matlab_vertex_points.txt', 'r') as matlab_file:
    # Read the lines of the files into lists
    sphere_lines = sphere_file.readlines()
    matlab_lines = matlab_file.readlines()

# Open the output file for writing
with open('matched_lines.txt', 'w') as output:
    # Loop through the lines of the output file
    for i, matlab_line in enumerate(matlab_lines):
        # Extract the x, y, z values from the output file line
        output_x, output_y, output_z = map(float, matlab_line.strip().split())

        # Loop through the lines of the sphere test file
        for j, sphere_line in enumerate(sphere_lines):
            # Extract the x, y, z values from the sphere test file line
            sphere_x, sphere_y, sphere_z, *_ = map(float, sphere_line.strip().split())

            # Compare the x, y, z values to 3 decimal places
            if round(output_x, 2) == round(sphere_x, 2) and \
               round(output_y, 2) == round(sphere_y, 2) and \
               round(output_z, 2) == round(sphere_z, 2):
                # Write the line numbers to the output file
                output.write(f'{i+1} {j+1}\n')
                break
