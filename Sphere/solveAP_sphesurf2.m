%% Set up
clear all
close all
% figure
name='SphericalSurface';

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

%% FEM model setup
APmodel=createpde(2);
% walls=niftiread('LV.nii');
% fac=2;
% walls=imresize3(walls,fac,'nearest');
% me=isosurface(walls,0.5);

% me=myreadmeshvtk2('walls3MC.vtk');
% gm=geometryFromMesh(APmodel,me.vertices,me.faces);

% hollow sphere
r1=10;
r2=12;
gm = multisphere([r1 r2],'Void',[true,false]);
% add vertex to use as initial condition
% addVertex(gm,'Coordinates',[0 0 12]);

APmodel.Geometry=gm;
pdegplot(APmodel,'CellLabels','on','FaceLabels','on','FaceAlpha',0);

hold all
mesh=generateMesh(APmodel);
nnodes=length(mesh.Nodes);
pdemesh(APmodel); 
axis equal

% ver=APmodel.Mesh.Nodes';
% tol2=0.001;
% indi1=find(sum(ver.^2,2)-r2.^2>tol2);
% indi2=find(sum(ver.^2,2)-r1.^2>tol2);
% ver(cat(1,indi1,indi2),:)=[];
% ver(882,:)=[];

% theta=0:0.1:pi;
% phi=0:0.2:2*pi;
% [phim,thetam]=meshgrid(phi,theta);
% [x1,y1,z1]=sph2cart(thetam,phim,r1);
% [x2,y2,z2]=sph2cart(thetam,phim,r2);
% ver=[cat(1,x1(:),x2(:)) cat(1,y1(:),y2(:)) cat(1,z1(:),z2(:))];

% ver=APmodel.Mesh.Nodes';
% k=[];
% ind=0;
% for i=1:nnodes
%     try
%         addVertex(gm,'Coordinates',ver(i,:));
%         k=cat(1,k,i);
%     catch
%         continue;
%     end
% end
% disp(['Added ' num2str(length(k)) ' vertices.']);
% disp(k)
% 
% hold all
% plot3(ver(:,1),ver(:,2),ver(:,3),'k.')
% vv2=addVertex(gm,'Coordinates',vver);

specifyCoefficients(APmodel,'m',coeffs.m,'d',coeffs.d, ...
    'c',coeffs.c,'a',coeffs.a,'f',coeffs.f);
applyBoundaryCondition(APmodel,'neumann','face',1:2,'g',[0;0],'q',[0;0]);
% ufun=@(location,~)location.z>8;
load forSphereSpiral.mat
setInitialConditions(APmodel,R);
% setInitialConditions(APmodel,ini(k,:)','Vertex',1:length(k));

% setInitialConditions(APmodel,[0 0]');
% 
% for i=1:length(k)
%     setInitialConditions(APmodel,[0.5 0.5]','Vertex',i);
% end
% use findnodes to set the other nodes on


%% Solve!
tini=0;
tfin=201+20;
dt=0.5;
dt_disp=2;
tlist = 201:dt:tfin;

APmodel.SolverOptions.RelativeTolerance = 1.0e-3; 
APmodel.SolverOptions.AbsoluteTolerance = 1.0e-4;

APmodel.SolverOptions.ReportStatistics='on';
R2 = solvepde(APmodel,tlist);
u = R2.NodalSolution;

% figure
% nod=1;
% plot(tlist,squeeze(u(nod,1,:)))
% hold all
% plot(tlist,squeeze(u(nod,2,:)))
% grid on
% xlabel('Time (s)')
% legend('V','W')

% figure; 
% tshow=30;
% plot(squeeze(u(:,1,tshow)),'.')
% hold all
% plot(squeeze(u(:,2,tshow)),'.')
% grid on
% xlabel('Node')
% legend('V','W')

% ylabel 'u_{heart} (AU)'

for t=tini+dt:dt:tfin
    if ~mod(t,dt_disp)
        mywritemeshvtktetra(mesh,squeeze(u(:,1,t)),['SphereLinear2/' ...
            name num2str(t,'%d') '.vtk']);
    end 
end
save([name '.mat'],'-v7')