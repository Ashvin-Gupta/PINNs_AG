% Script to write the Vsav variable to a text file
% Requires the vairbale Vsav, which is produced upon running a maltab
% generation file
% Assuming Vsav is a variable containing data

file_path = 'Vsav.txt';

% Open the file in write mode
file_id = fopen(file_path, 'w');

% Check if the file was opened successfully
if file_id == -1
    error('Failed to open the file for writing.');
end

% Write the variable data to the file
fprintf(file_id, '%f\n', Vsav);

% Close the file
fclose(file_id);

disp('Variable saved as a text file successfully.');
