% function [Vsav,Wsav]=AlievPanfilov2D_RK_Istim_heter_spiral(tfin,ncells,iscyclic,flagmovie,Dfac)
% (Ventricular) Aliev-Panfilov model in single-cell with the parameters
% reparameterisation using parameters in Panfilov, Phys Rev Letters, 2022
% doi.org/10.1103/PhysRevLett.88.118101
% Marta, 23/04/2021

% tend is the time (AU) at which the simulation terminates
% ncells is number of cells in 1D cable (e.g. 200)
% iscyclic, = 0 for a cable, = 1 for a ring (connecting the ends of the
% cable - the boundary conditions are not set for the ring yet!)
% flagmovie, = 0 to show a movie of the potential propagating, = 0
% otherwise
% Dfac is the reduction factor ]0,1] of D in the (rectagular) heterogeneity
% set to 1 for no heterogeneity (setting to 0 for a hole is not
% recommended, as the boundary conditions do not account for this)


% Aliev-Panfilov model parameters 
% V is the electrical potential difference across the cell membrane in 
% arbitrary units (AU)
% t is the time in AU - to scale do tms = t *12.9

close all
clear all
tstar=355; % time, AU, at which the data starts being saved (because the 
% spiral wave has formed already)
tend = 150+tstar;
tCF=67; % time (AU) at which the extra stimulus is applied
tCF2=100;
tCF3=150;
% Dfac = 1; % factor by which D0 is reduced in fibrotic area
% extra=0;
ncells_x=180;
ncells_y=180;
iscyclic=0;
flagmovie=1;
matname='Spiral_Marta_v2';

index  = 1;

setglobs(ncells_x, ncells_y);
global h

% one of the biggest determinants of the propagation speed
% (D should lead to realistic conduction velocities, i.e.
% between 0.6 and 0.9 m/s)
X = ncells_x + 2; % to allow boundary conditions implementation
Y = ncells_y + 2;
stimgeo=false(X,Y);
stimgeo(1:5,:)=true; % indices of cells where external stimulus is felt

crossfgeo=false(X,Y); % extra stimulus to generate spiral wave
crossfgeo(:,1:floor(X/2.5))=true;

crossfgeo2=false(X,Y); % extra stimulus to generate spiral wave
crossfgeo2(20:40,20:40)=true;

crossfgeo3=false(X,Y); % extra stimulus to generate spiral wave
crossfgeo3(70:90,70:90)=true;

% Model parameterse
% time step below 10x largr than for forward Euler
dt=0.005; % AU, time sund(1/dt); % number of iterations at which V is outputted
% for plotting, set to correspond to 1 ms, regardless of dt
% tend=BCL*ncyc+extra; % AU, duration of simulation
%tep for finite differences solver
dt=0.005; % AU, time step for finite differences solver
gathert=600; %ro
stimdur=2; % AU, duration of stimulus
Ia=0.2; % AU, value for Istim when cell is stimulated

V(1:X,1:Y)=0; % initial V
W(1:X,1:Y)=0.01; % initial W

Vsav=zeros(ncells_x,ncells_y,ceil((tend-tstar)/gathert)); % array where V will be saved during simulation
Wsav=zeros(ncells_x,ncells_y,ceil((tend-tstar)/gathert)); % array where W will be saved during simulation


ind=0; %iterations counter

yfun=zeros(2,size(V,1),size(V,2));

% for loop for explicit RK4 finite differences simulation
for t=dt:dt:tend % for every timestep
    ind=ind+1; % count interations
        % stimulate at every BCL time interval for ncyc times
        if t<=stimdur
            Istim=Ia*stimgeo; % stimulating current
        elseif t>=tCF&&t<=tCF+stimdur
            Istim=Ia*crossfgeo;
%         elseif t>=tCF2&&t<=tCF2+stimdur
%             Istim=Ia*crossfgeo2;
%         elseif t>=tCF3&&t<=tCF3+stimdur
%             Istim=Ia*crossfgeo3;
        else
            Istim=zeros(X,Y); % stimulating current
        end
        
        % 4-step explicit Runga-Kutta implementation
        yfun(1,:,:)=V;
        yfun(2,:,:)=W;
        k1=AlPan(yfun,Istim);
        k2=AlPan(yfun+dt/2.*k1,Istim);
        k3=AlPan(yfun+dt/2.*k2,Istim);
        k4=AlPan(yfun+dt.*k3,Istim);
        yfun=yfun+dt/6.*(k1+2*k2+2*k3+k4);
        V=squeeze(yfun(1,:,:));
        W=squeeze(yfun(2,:,:));
                      
        % rectangular boundary conditions: no flux of V
        V(1,:)=V(2,:);
        V(end,:)=V(end-1,:);
        V(:,1)=V(:,2);
        V(:,end)=V(:,end-1);

        
        % At every gathert iterations, save V value for plotting
        if t>=tstar&&mod(ind,gathert)==0
            % save values
            Vsav(:,:,index)=V(2:end-1,2:end-1);
            Wsav(:,:,index)=W(2:end-1,2:end-1);
            index = index +1;
            % show (thicker) cable
            if flagmovie
                subplot(2,1,1)
                imagesc(V(2:end-1,2:end-1)',[0 1])
                hold all
%                 if Dfac<1 % there is a heterogeneity
%                     rectangle('Position',[fibloc(1) fibloc(1) ...
%                         fibloc(2)-fibloc(1) fibloc(2)-fibloc(1)]);
%                 end
                axis image
                set(gca,'FontSize',14)
                xlabel('x (voxels)')
                ylabel('yfun (voxels)')
                set(gca,'FontSize',14)
                title(['V (AU) - Time: ' num2str(t,'%.0f') ' ms'])
                colorbar
                hold off
                
                subplot(2,1,2)
                imagesc(W(2:end-1,2:end-1)',[0 1])
                hold all
%                 if Dfac<1 % there is a heterogeneity
%                     rectangle('Position',[fibloc(1) fibloc(1) ...
%                         fibloc(2)-fibloc(1) fibloc(2)-fibloc(1)]);
%                 end
                axis image
                set(gca,'FontSize',14)
                xlabel('x (voxels)')
                ylabel('yfun (voxels)')
                set(gca,'FontSize',14)
                title(['V (AU) - Time: ' num2str(t,'%.0f') ' ms'])
                colorbar
                set(gca,'FontSize',14)
                title('W (AU)')
                colorbar
                pause(0.01)
                hold off
%                 waitforbuttonpress;
            end
        end
end

x=h:h:ncells_x*h;
y=h:h:ncells_y*h;
t=1:1:size(Vsav,3);
%save([matname '.mat'],'Vsav','Wsav','x','y','t','-v7')

function dydt = AlPan(yfun,Istim)
    global a k mu1 mu2 epsi b h D
    
    V=squeeze(yfun(1,:,:));
    W=squeeze(yfun(2,:,:));
    
    [gx,gy]=gradient(V,h);
    [Dx,Dy]=gradient(D,h);
    
    dV=4*D.*del2(V,h)+Dx.*gx+Dy.*gy; % extra terms to account for heterogeneous D
    dWdt=(epsi + mu1.*W./(mu2+V)).*(-W-k.*V.*(V-b-1));
    dVdt=(-k.*V.*(V-a).*(V-1)-W.*V)+dV+Istim;
    dydt(1,:,:)=dVdt;
    dydt(2,:,:)=dWdt;
end

% function setglobs(ncells,Dfac)
function setglobs(ncells_x, ncells_y)
    global a k mu1 mu2 epsi b h D X Y
    a = 0.1; %0.01;
    k = 8; %10;
    mu1 = 0.06; 
    mu2 = 0.3;
    epsi = 0.01; % 0.002;
    b  = 0.1; %0.08; 
    h = 0.15; % mm cell length
    X = ncells_x + 2; % to allow boundary conditions implementation
    Y = ncells_y + 2; 
%     fibloc=[floor(X/3) ceil(X/3+X/5)]; % location of (square) heterogeneity
    
    D0 = 0.15; % mm^2/UA, diffusion coefficient (for monodomain equation)

    D = D0*ones(X,Y);
%     D(fibloc(1):fibloc(2),fibloc(1):fibloc(2))=D0*Dfac;
end


% end