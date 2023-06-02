function usav=loadFK_2D(loadnam,moviename,nms)
% Fenton-Karma model in 2D
% same implementation as in FentonKarma2.m, following setup in June3_in2D.m
% Marta, 19/11/2015
% adapted on 21/02/2023

close all
% clear all

% set up video saving
figure
if exist('moviename','var')
    if ~isstring(moviename)
        moviename=num2str(moviename);
    end
    writerObj = VideoWriter(strcat(moviename,'.avi'));
    writerObj.FrameRate = 3;
    open(writerObj);
end

% load state variables so that a rotor is present initially
% from iniFK_2D.m
load(loadnam);

X=pars.X;
Y=pars.Y;
D=pars.D;
dt=pars.dt;
gathert=1e4; % every 100 ms
savet=100; % every ms

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

% load data
v=iniv;
w=iniw; 
u=iniu; 

ind=0;
savind=0;

% initialise saved variables
usav=zeros(X-2,Y-2,nms);
% wsav=usav;
% vsav=usav;
for ms=dt:dt:nms % time loop
        ind=ind+1; % integer counter

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
        if exist('moviename','var')
            if mod(ind,gathert)==0
                imagesc(u(2:end-1,2:end-1),[0 1])
                colorbar
                text(5,5,['t:' num2str(ms) ' ms'],'color','k')
                disp(['t:' num2str(ms) ' ms'])         
                frame = getframe;
                writeVideo(writerObj,frame);
            end
        end
        
        if mod(ind,savet)==0
            savind=savind+1;
            usav(:,:,savind)=u(2:end-1,2:end-1);
%             wsav(:,:,savind)=w(2:end-1,2:end-1);
%             vsav(:,:,savind)=v(2:end-1,2:end-1);
        end

 end % end of ms loop

 if exist('moviename','var')
    close(writerObj);   
 end

% save variables
% save(moviename,'usav','vsav','wsav','pars','-v7.3'); 
save(moviename,'usav','pars','-v7.3'); 
disp('End.');