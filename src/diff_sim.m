function S=diff_sim(I,seq,simu)

% This function computes the diffusion MRI signal provided the 2-3 
% dimensional substrate I, the sequence seq, and simulation parameters
% simu.
%
% Inputs (n-dimensional substrate):
%
%   + I: substrate mask
%   + seq: structure containing the following fields
%         - G: normalised gradient sequence (Np x n) 
%         - t: time discretisation used for G (Np x n)
%         - G_r: gradient strengths (1 x Ng)
%   + simu: structure containing the following fields
%         - D: diffusion constants
%         - N: number of diffusion particles
%         - r: resolution of I, i.e. pixel size (1 x n)

%% TO DO:
% permeability
% relaxivities

%% Extract parameters and define basic constants

t=seq.t;G=seq.G;
D=simu.D
N=simu.N;
r=simu.r;

dt=t(2)-t(1);
gam=2.675e8;

N_i=size(I);
dim=numel(N_i);
dx=sqrt(2*dim*D*dt);

N_c=sum(unique(I(:))>0); % number of diffusion compartments
if N_c>1,P=simu.P;else P=NaN;end

% set number of particles to "throw in"
N=round(N*numel(I)/sum(I(:)>0));


%% Do

% distribute points troughout the domain
%X=prodsum(rand(N,dim),r.*N_i);
X=(rand(N,dim)) .* (r.*N_i);
ind_s=mask_pos(X,N_i,r);

X(I(ind_s)==0,:)=[]; ind_s(I(ind_s)==0)=[]; % delete points in non-diffusion regions
N_p=size(X,1); % number of resulting particles
%Xn = zeros(N_p, dim);
%pre_ang = zeros(N_p, 1);
pre_dx = zeros(N_p, 1) + dx;

phase=zeros(N_p,1);


% Xa=zeros(length(t),2);
for tt=1:length(t)
    
    %Xn=prodsum(rand_circ(N_p,dim),dx(I(ind_s))); % new proposed step
    if dim==2
        pre_ang = rand(N_p,1)*(2*pi);
        [Xn(:,1),Xn(:,2)] = pol2cart(pre_ang, pre_dx);
    elseif dim==3
        pre_ang_rho = rand(N_p,1)*2*pi;
        pre_ang_phi = rand(N_p,1)*2*pi;
        [Xn(:,1),Xn(:,2),Xn(:,3)] = sph2cart(pre_ang_rho, pre_ang_phi, pre_dx);
    end
    ind_p=mask_pos(mod(X+Xn,N_i(1)*r),N_i,r); % assuming a square image
    Ie=I(ind_s)==I(ind_p);
    
    % permeability (NOT TESTED)
    if sum(~isnan(P(:)))>0
        Ip=find(and(~Ie,I(ind_p)~=0));
        ind_k=sub2ind([N_c,N_c],I(ind_s(Ip)),I(ind_p(Ip))); % find compartments between moving points
        prob=r(1)/2*P(ind_k)*2./D(I(ind_s(Ip)))>rand(length(ind_k),1);
        ind_s(Ip(prob))=ind_p(Ip(prob));
        Ie=[find(Ie);Ip(prob)];
    end
    if dim==2
        Xn(:,1) = Xn(:,1).*Ie; Xn(:,2) = Xn(:,2).*Ie;
    elseif dim==3
        Xn(:,1) = Xn(:,1).*Ie; Xn(:,2) = Xn(:,2).*Ie; Xn(:,3) = Xn(:,3).*Ie;
    end
    X=X+Xn; % move points 
    %X(Ie,:)=X(Ie,:)+Xn(Ie,:); % move points 
    
    phase=phase+sum(X .* G(tt,:),2); % update phase
    
 
    % apply periodic BCs
    M1=rem(X,N_i(1)*r);  
    M2=mod(X,N_i(1)*r);
    Ind_n= ~(M2==X).*sign(M1).*Ie;
    phase=phase+sum(-Ind_n*N_i(1)*r .* sum(G(1:tt,:)),2);
    X=M2;
%     X(Ie,:)=X(Ie,:)-Ind_n*N_i(1)*r;
    
    
    %Xa(tt,:)=X(3000,:);
    
    disp(tt)
end

% [Xx,Yy]=meshgrid(1:N_i(1));
% figure,pcolor(Xx*r,Yy*r,I),shading interp,hold on,plot(Xa(:,1),Xa(:,2),'.-')

if ~isfield(seq,'G_s'),seq.G_s=1;end
S=zeros(length(seq.G_s),1);
for gg=1:length(S),S(gg)=mean(exp(1j*gam*dt*phase*seq.G_s(gg)));end

end

%%
% function A=prodsum(A,B)
%     A = A .* B;
% end
% 
% 
% %% 
% function s=rand_circ(N,dim)
% 
% % As far as I can tell this is actually an allocating function? Both U and
% % s should be possible to preallocate, even though it will congest the
% % computation
% 
% if dim==2
%     U=rand(N,1);
%     s=[cos(2*pi*U),sin(2*pi*U)];
% elseif dim==3
%     fi=2*pi*rand(N,1);
%     ti=acos(1-2*rand(N,1));
%     s=[sin(ti).*cos(fi),sin(ti).*sin(fi),cos(ti)];
% end
% 
% end

%% 
function ind_s=mask_pos(X,N_i,r)

% This function computes the voxel in I where each particle in X belongs to.

X=ceil(X .* 1./r);

X(X<1)=1;
X(X(:,1)>N_i(1),1)=N_i(1);
X(X(:,2)>N_i(2),2)=N_i(2);

if length(N_i)==2
    %ind_s=sub2ind(N_i,X(:,1),X(:,2));
    ind_s = X(:,1) + (X(:,2)-1)*N_i(1);
elseif length(N_i)==3
    X(X(:,3)>N_i(3),3)=N_i(3);
    ind_s=sub2ind(N_i,X(:,1),X(:,2),X(:,3));
end

end



