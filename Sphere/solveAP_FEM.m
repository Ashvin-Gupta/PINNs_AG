%% Set up
clear all
close all
% figure
name='APtest';

% Model Parameters
% aa = 0.01;
% k = 8.0;
% mu1 = 0.2;
% mu2 = 0.3;
% epsi = 0.002;
% b  = 0.15;
% D = 0.1;

% Symbolic PDE Definition
syms W(x,y,z,t) V(x,y,z,t)
syms aa k mu1 mu2 epsi b D
fV = k*V*(V-aa)*(V-1)+V*W; 
fW = (epsi + mu1*W/(V+mu2))*(-W-k*V*(V-b-1));
pdeeq = [diff(V,t) - D*laplacian(V,[x,y,z]) + fV; ...
    diff(W,t) - fW];

symCoeffs = pdeCoefficients(pdeeq,[V;W],'Symbolic',true);
symVars = [aa k mu1 mu2 epsi b D];
symCoeffs = subs(symCoeffs, symVars, [0.01 8.0 0.2 0.3 0.002 0.15 0.1]);
coeffs = pdeCoefficientsToDoubleMV(symCoeffs);
coeffs.f = zeros(2,1,'double'); % make this explicit to avoid crashing solver!!!

% FEM model setup
APmodel=createpde(2);
gm=multicylinder([4 4.2],[6 6],'Void',[true,false]);
APmodel.Geometry=gm;
pdegplot(APmodel,'CellLabels','on','FaceLabels','on','FaceAlpha',0.3);

hold all
mesh=generateMesh(APmodel);
pdemesh(APmodel); 
axis equal

specifyCoefficients(APmodel,'m',coeffs.m,'d',coeffs.d, ...
    'c',coeffs.c,'a',coeffs.a,'f',coeffs.f);
applyBoundaryCondition(APmodel,'neumann','face',1:4,'g',[0;0;0;0],'q',[0;0;0;0]);
% ufun=@(location,~)location.z>8;
setInitialConditions(APmodel,[0;0]);
setInitialConditions(APmodel,[0.8;0],'face',1);

%% Solve!
tini=0;
tfin=100;
dt=0.5;
dt_disp=2;
tlist = tini:dt:tfin;

APmodel.SolverOptions.RelativeTolerance = 1.0e-3; 
APmodel.SolverOptions.AbsoluteTolerance = 1.0e-4;

APmodel.SolverOptions.ReportStatistics='on';
R = solvepde(APmodel,tlist);
u = R.NodalSolution;

figure
nod=1;
plot(tlist,squeeze(u(nod,1,:)))
hold all
plot(tlist,squeeze(u(nod,2,:)))
grid on
xlabel('Time (s)')
legend('V','W')

figure; 
tshow=30;
plot(squeeze(u(:,1,tshow)),'.')
hold all
plot(squeeze(u(:,2,tshow)),'.')
grid on
xlabel('Node')
legend('V','W')

% ylabel 'u_{heart} (AU)'

for t=tini+dt:dt:tfin
    if ~mod(t,dt_disp)
        mywritemeshvtktetra(mesh,squeeze(u(:,1,t))*100-80,[name num2str(t,'%.0f') '.vtk']);
    end 
end
