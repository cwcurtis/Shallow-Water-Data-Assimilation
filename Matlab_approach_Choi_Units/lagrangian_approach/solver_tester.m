function solver_tester(K,Llx,tf,dt)

KT = 2*K;
ep = .1;
mu = sqrt(ep);
Mval = 14;
dts = 1;
Nens = 1;
sig = 1;

params = [K ep mu Llx Mval dt dts Nens sig];
Xfloats = [-3 -1 1 3]';
Ndat = length(Xfloats);
xint = zeros(2*Ndat,1);
xint(1:2:2*Ndat-1) = Xfloats;
xint(2:2:2*Ndat) = ep*cos(pi*Xfloats/Llx);
nmax = round(tf/dt);

Xmesh = linspace(-Llx,Llx,KT+1);
Xmesh = Xmesh(1:KT)';

Kmesh = pi/Llx*[0:K -K+1:-1]';
mDk = mu*Kmesh;
tnh = tanh(mDk);
L1 = Kmesh.*tnh/mu;

eta0 = cos(pi*Xmesh/Llx);
q0 = sin(pi*Xmesh/Llx);
un = [eta0;q0;xint];

Zs = zeros(KT);
Is = eye(KT);

Lop = [Zs diag(L1);-Is  Zs];

eLdt2 = expm(dt*Lop/2.);
eLdt = eLdt2*eLdt2;

path_dat = zeros(2*Ndat,nmax);
surf_dat = zeros(KT,nmax);

% Build "true" data paths
for kk=1:nmax   
    path_dat(:,kk) = un(2*KT+1:end);
    surf_dat(:,kk) = un(1:KT);    
    un = solver_rk4(params, L1, Kmesh, mDk, tnh, eLdt, eLdt2, un, Ndat);        
end

dx = Llx/K;
Xvals = (-Llx:dx:Llx-dx)';
clf
hold on
plot(path_dat(1,:),path_dat(2,:),'--')
plot(Xvals,ep*surf_dat(:,nmax),'k')
hold off