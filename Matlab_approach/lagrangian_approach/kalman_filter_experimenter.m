function [fin_dat,tvals,msqerror] = kalman_filter_experimenter(K,Llx,tf,dt,dts,Nens,sig,Xfloats)

KT = 2*K;
ep = .1;
mu = sqrt(ep);
Mval = 14;

params = [K ep mu Llx Mval dt dts Nens sig];
Ndat = length(Xfloats);
xint = zeros(2*Ndat,1);
xint(1:2:2*Ndat-1) = Xfloats;
xint(2:2:2*Ndat) = ep*cos(pi*Xfloats/Llx);
nindt = round(dts/dt);
nmax = round(tf/dt);
nsamp = round(nmax/nindt);

Xmesh = linspace(-Llx,Llx,KT+1);
Xmesh = Xmesh(1:KT)';

Kmesh = pi/Llx*[ 0:K -K+1:-1 ]';
L1 = Kmesh.*tanh(mu.*Kmesh)/mu;
mDk = mu*Kmesh;
tnh = tanh(mDk);

eta0 = cos(pi*Xmesh/Llx);
q0 = sin(pi*Xmesh/Llx);
un = [eta0;q0;xint];

Zs = zeros(KT);
Is = eye(KT);

Lop = [Zs diag(L1);-Is  Zs];

eLdt2 = expm(dt*Lop/2.);
eLdt = eLdt2*eLdt2;

eLdt2s = expm(dts*Lop/2.);
eLdts = eLdt2s*eLdt2s;

path_dat = zeros(2*Ndat,nsamp);
surf_dat = zeros(KT,nsamp);

% Build initial ensemble of forecasts
% First, use what interpolatory data we have.   
etames = cos(pi*Xfloats/Llx) + sig*randn(length(Xfloats),1);
path_dat(1:2:2*Ndat-1,1) = Xfloats;
path_dat(2:2:2*Ndat,1) = ep*cos(pi*Xfloats/Llx);
surf_dat(:,1) = eta0;
etaemp = interpft(etames,KT);
xf = repmat([etaemp;zeros(KT,1);xint],1,Nens) + [zeros(KT,Nens);sig*randn(KT+2*Ndat,Nens)];

% Build "true" data paths
dcnt = 2;
for kk=1:nmax
    params(5) = 14;
    un = solver_rk4(params, L1, Kmesh, mDk, tnh, eLdt, eLdt2, un, Ndat);
    
    if mod(kk,nindt) == 0
       path_dat(:,dcnt) = un(2*KT+1:end) + sig*randn(2*Ndat,1);
       surf_dat(:,dcnt) = un(1:KT);
       dcnt = dcnt + 1;
    end
    
end

params(5) = Mval;
msqerror = zeros(nsamp);
msqerror(1) = norm(interpft(etames,KT) - eta0)/norm(eta0);
tvals = zeros(nsamp);

for jj = 2:nsamp
    rmat =  sig*randn(2*Ndat,Nens);
    dmat = repmat(path_dat(:,jj),1,Nens) + rmat;
    cormat = rmat*rmat';
    
    xa = analysis_step(K,Nens,xf,dmat,cormat);
    xapprox = sum(xa(1:KT,:),2)/Nens;
    msqerror(jj) = norm(xapprox-surf_dat(:,jj))/norm(surf_dat(:,jj));
    tvals(jj) = dt*jj*nindt;
    
    if jj<nsamp
        parfor kk=1:Nens
            xf(:,kk) = solver_rk4(params,L1,Kmesh,mDk,tnh,eLdts,eLdt2s,xa(:,kk),Ndat);
        end
    end
end        
               
xapprox = sum(xf(1:KT,:),2)/Nens;
fin_dat = [xapprox;surf_dat(:,nsamp)];