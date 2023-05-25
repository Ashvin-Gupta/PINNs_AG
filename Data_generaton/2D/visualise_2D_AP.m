% Set the filename of the text file containing the data
close all
clear all
filename = 'v_pred_model_spiral_future.txt';

% Open the file and read in the data
fileID = fopen(filename);
data = fscanf(fileID, '%f %f %f %f', [4, Inf]);
fclose(fileID);

% Extract the x, y, time, and u data from the matrix
x = data(1,:);
y = data(2,:);
time = data(3,:);
u = data(4,:);

% Determine the unique time points in the data
unique_time = unique(time);

% Determine the number of time points in the data
num_time = length(unique_time);

% Determine the number of spatial points in the data
num_space = length(x) / num_time;

% Reshape the x, y, and u data into 2D matrices
num_rows = length(unique(y));
num_cols = length(unique(x));

u_mat = reshape(u, [num_rows, num_cols, num_time]);

% Create a new figure
figure;

% Initialize the movie object
M = struct('cdata', cell(1, size(u_mat, 3)), 'colormap', cell(1, size(u_mat, 3)));

% Loop through each time point and plot the corresponding u values
for t = 1:size(u_mat, 3)
    % Clear the previous plot
    clf;
    
    % Plot the u values for the current time point
    pcolor(u_mat(:,:,t));
    shading interp;
    colorbar;
    
    % Set the title to display the current time
    title(sprintf('Time: %.2f', t));
    
    % Save the current frame to the movie object
    M(t) = getframe;
    pause(0.01)
end

% Play the movie
movie(M);


