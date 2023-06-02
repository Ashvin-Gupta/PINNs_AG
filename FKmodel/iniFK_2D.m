function [iniu,iniv,iniw]=iniFK_2D(savnam,flagmovie)
% Fenton-Karma model in 2D
% Marta, 19/11/2015
% adapted on 21/02/2023

close all

% call parameter setting file if not provided
pars=fun_setpars();

if ~exist('savnam','var')
elseif isempty(savnam)
    savnam='test';
end

figure
% save animation frames as movie if requested
if exist('flagmovie','var')
    if flagmovie>0
        moviename=savnam;
        writerObj = VideoWriter(strcat(moviename,'.avi'));
        open(writerObj);
    end
end

% read parameters from file
X=pars.X;
Y=pars.Y;
pacegeo=pars.pacegeo;
crossgeo=pars.crossgeo;
D=pars.D;
dt=pars.dt;
gathert=pars.gathert;
crosstime=pars.crosstime;
stimdur=pars.stimdur;
nms=pars.nms;

% read model parameters from .mat file
load FKpars2023.mat

% use Goodman set 1 (atrial, remodelled)
if contains(pars.model,'Goodman')
    n=str2num(pars.model(end));
    uv=FKGoodman(11,n);
    uw=uv;
    uu=uv;
    uvsi=FKGoodman(12,n);
    ucsi=FKGoodman(13,n);
    k=FKGoodman(2,n);
    taud=FKGoodman(1,n);
    tauv2=FKGoodman(8,n);
    tauv1=FKGoodman(7,n);
    tauvplus=FKGoodman(6,n);
    tauo=FKGoodman(5,n);
    tauwminus=FKGoodman(10,n);
    tauwplus=FKGoodman(9,n);
    taur=FKGoodman(3,n);
    tausi=FKGoodman(4,n);
end

% uv=0.160; % uc for v
% uw=0.160; % uc for w
% uu=0.160; % uc for u
% uvsi=0.040; % uv
% ucsi=0.85; %uc_si 
% k=10; % k
% taud=0.125; % tau_d
% tauv2=60; % tauv2-
% tauv1=82.5; % tauv1-
% tauvplus=5.75; % tauv+
% tauo=32.5; 
% tauwminus=400; % tauw- 
% tauwplus=300; % tauw+ 
% taur=70; % taur
% tausi=114; % tausi

% initial conditions
ua=0.95; % value for u of initially activated cells
v=0.99*ones(X,Y); % v
w=0.99*ones(X,Y); % w
u=zeros(X,Y);

ind=0;
for ms=dt:dt:nms % time loop
        ind=ind+1; % integer counter

        % stimulation protocol
        % initial stimulus
        if ms<stimdur
            u(pacegeo==1)=ua;
        end
        % second stimulus for spiral
        if abs(ms-crosstime)<dt
            u(crossgeo==1)=0;
            v(crossgeo==1)=0.99;
            w(crossgeo==1)=0.99;        
        end
        
        %%% Fenton Karma Model %%%

        %fast inward current and gate
        Fu=zeros(X,Y);
        vinf=ones(X,Y);
        tauv=tauvplus*ones(X,Y);
        
        vinf(u>=uv)=0;
        tauv(u<uvsi&u<uv)=tauv1;
        tauv(u>=uvsi&u<uv)=tauv2;
        Fu(u>=uv)=(u(u>=uv)-uv).*(1-u(u>=uv));

        %fast inward current
        Jfi=Fu.*(-v)./taud;

        %update v
        v=v+(vinf-v)./tauv.*dt;

        %ungated outward current
        Uu=ones(X,Y);
        Uu(u<=uu)=u(u<=uu);
        tauu=taur*ones(X,Y);
        tauu(u<=uu)=tauo;

        Jso=Uu./tauu;

        %slow inward current and slow gate
        winf=ones(X,Y);
        winf(u>=uw)=0;
        tauw=tauwminus*ones(X,Y);
        tauw(u>=uw)=tauwplus;
        Jsi=-w./tausi.*0.5.*(1+tanh(k.*(u-ucsi)));

        % update w
        w=w+(winf-w)./tauw*dt;

        Iion=-(Jfi+Jsi+Jso);
        u=u+(Iion+D.*del2(u)).*dt;
                
        % boundary conditions (Dirichlet)
        u(1,:)=u(2,:);
        u(end,:)=u(end-1,:);
        u(:,1)=u(:,2);
        u(:,end)=u(:,end-1);
        
        % show images
        if mod(ind,gathert)==0
            imagesc(u(2:end-1,2:end-1),[0 1])
            colorbar
            text(5,5,['t:' num2str(ms) ' ms'],'color','k')
            disp(['t:' num2str(ms) ' ms']) 
            pause(0.01);
            if exist('moviename','var')
                fram = getframe;
                writeVideo(writerObj,fram);
            end
        end
        
 end % end of ms loop
 if exist('flagmovie','var')
    if flagmovie>0
        close(writerObj);  
    end
 end
disp('End.');

% save final state for future simulation
iniv=v;
iniw=w;
iniu=u;

if exist('savnam','var')
    save(savnam,'iniv','iniw','iniu','pars','-v7');
end
