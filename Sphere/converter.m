% Open the file
fileID = fopen('v_pred_model.txt','r');

% Read the data into a matrix
data = fscanf(fileID,'%f %f %f %f %f',[5 Inf]);
data = data'; % Transpose so that each row corresponds to a time point

% Close the file
fclose(fileID);

% Extract the unique time points
time = unique(data(:,4));
num_time_points = numel(time);

% Find the number of U values per time point
num_U_values = sum(data(:,4) == time(1));

% Preallocate the U_data matrix
U_data = zeros(num_U_values, num_time_points);

% Populate the U_data matrix
for i = 1:num_time_points
    idx = data(:,4) == time(i);
    U_data(:,i) = data(idx,5);
end

%%
for t=tini+dt:dt:tfin
    if ~mod(t,dt_disp)
        mywritemeshvtktetra(mesh,squeeze(U_data(:,t))*100-80,[name num2str(t,'%.0f') '.vtk']);
    end 
end
save([name '.mat'],'-v7')

%% Get similar indicie values 
 
array1 = mesh.Nodes(1,:);
array2 = data(1:7137);

common_nums = [];
array1_positions = [];
array2_positions = [];

% Loop through each element in array1 and check if it is present in array2
for i = 1:length(array1)
    if ismember(round(array1(i),14), round(array2,14))
        % If the element is present in array2, then store its position in both arrays
        common_nums = [common_nums, array1(i)];
        array1_positions = [array1_positions, i];
        array2_positions = [array2_positions, find(round(array2,14)==round(array1(i),14))];
    end
end

%% Plot random indicies 

matlab = [mesh.Nodes];
pinns = [data(1:7137,1), data(1:7137,2), data(1:7137,3) ];
pinns = pinns';

rand_indices_1 = randperm(size(matlab, 2), 1000);
rand_indices_2 = randperm(size(pinns, 2), 1000);


rand_points_1 = matlab(:, rand_indices_1);
rand_points_2 = pinns(:, rand_indices_1);

%rand_points_1 = matlab(:, array1_positions);

% Plot the two arrays against each other in 3D
scatter3(rand_points_1(1,:), rand_points_1(2,:), rand_points_1(3,:));
hold on;
scatter3(rand_points_2(1,:), rand_points_2(2,:), rand_points_2(3,:));
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Random points from Array 1 and Array 2 in 3D');
legend('matlab', 'pinns');



