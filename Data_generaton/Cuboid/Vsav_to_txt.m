% Script to write the Vsav variable to a text file
% Requires the vairbale Vsav, which is produced upon running a maltab
% generation file
% Assuming Vsav is a variable containing data

file_path = 'Vsav_3D_spiral_input.txt';

% Open the file in write mode
file_id = fopen(file_path, 'w');

% Check if the file was opened successfully
if file_id == -1
    error('Failed to open the file for writing.');
end
% Get the size of Vsav
[x_size, y_size, z_size, t_size] = size(Vsav);

% Write the variable data to the file
for t = 1:t_size
    for x = 1:x_size
        for y = 1:y_size
            for z = 1:z_size
                fprintf(file_id, '%f\n', Vsav(x, y, z, t));
            end
        end
    end
end

% Close the file
fclose(file_id);

disp('Variable saved as a text file successfully.');
