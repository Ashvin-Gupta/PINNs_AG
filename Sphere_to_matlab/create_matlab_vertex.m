array = load('mat_coord.mat','mat_coord');
coord = array.mat_coord';

fid = fopen('matlab_vertex_points.txt', 'w');

for i = 1:7095
    % Loop through each field and write it to the output file
    for j = 1:3
        % Get the value of the current field for the current element
       
        
        value = coord(i,j);

        % Write the value to the output file
        fprintf(fid, '%f ', value);
    end
    
    % Write a newline character to the output file to separate the elements
    fprintf(fid, '\n');
end

% Close the output file
fclose(fid);
