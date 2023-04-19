load('u.mat')
data = readmatrix('v_pred_model.txt');

vertex_indices = readmatrix('matched_lines.txt');

u_new = zeros(7095,201);

count = 0;

% Iterate over each time point
for t = 1:size(u, 3)
    % Set the vertices in vertex_indices to the corresponding U value
    for i = 1:size(vertex_indices, 1)
        vertex_index = vertex_indices(i, 1);
        row_index = vertex_indices(i, 2);
        u_new(vertex_index, t) = u(vertex_index, t);
        if ismember(row_index, 1:size(data, 1))
            u_new(vertex_index, t) = data(row_index + (7137*count), 5);
        end
    end

    % Set the remaining vertices to 0
    for vertex_index = 1:size(u_new, 1)
        if ~ismember(vertex_index, vertex_indices(:, 1))
            u_new(vertex_index, t) = 0;
        end
    end
    count  = count+1;
end

%%
tini=0;
tfin=100;
dt=0.5;

for t=tini+dt:dt:tfin
    if ~mod(t,dt_disp)
        mywritemeshvtktetra(mesh,squeeze(u_new(:,t))*100-80,[name num2str(t,'%.0f') '.vtk']);
    end 
end