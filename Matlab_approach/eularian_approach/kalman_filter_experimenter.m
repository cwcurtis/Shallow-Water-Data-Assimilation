function [fin_dat,tvals,msqerror] = kalman_filter_experimenter(K,Llx,tf,dt,dts,Nens,sig,Xfloats)

KT = 2*K;
Kc = floor(KT/3);
Kuc = KT - Kc + 1;
Kc = Kc + 1;
    
ep = .1;
mu = sqrt(ep);
Mval = 1;

params = [K ep mu Llx Mval dt dts Nens sig];
Ndat = length(Xfloats);
nindt = round(dts/dt);
nmax = round(tf/dt);
nsamp = round(nmax/nindt);

Hmat = ones(Ndat,KT-1);
evec = exp(1i*pi*Xfloats/Llx);

for jj=2:K   
    Hmat(:,jj) = Hmat(:,jj-1).*evec;
end

tmat = -2*imag(Hmat(:,2:K));
Hmat(:,2:K) = 2*real(Hmat(:,2:K));
Hmat(:,K+1:KT-1) = tmat;
Hmat = -1/KT*Hmat;

Xmesh = linspace(-Llx,Llx,KT+1);
Xmesh = Xmesh(1:KT)';

Kmesh = pi/Llx*[ 0:K -K+1:-1 ]';
L1 = Kmesh.*tanh(mu.*Kmesh)/mu;
mDk = mu*Kmesh;
tnh = tanh(mDk);

eta0 = cos(pi*Xmesh/Llx);
q0 = sin(pi*Xmesh/Llx);
etaf = fft(eta0);
un = [real(etaf(1:K));imag(etaf(2:K));q0];

Zs = zeros(KT);
Is = eye(KT);

Lop = [Zs diag(L1);-Is  Zs];

eLdt2 = expm(dt*Lop/2.);
eLdt = eLdt2*eLdt2;

eLdt2s = expm(dts*Lop/2.);
eLdts = eLdt2s*eLdt2s;

path_dat = zeros(Ndat,nsamp);
surf_dat = zeros(KT,nsamp);

% Build initial ensemble of forecasts
% First, use what interpolatory data we have.   
etames = cos(pi*Xfloats/Llx) + sig*randn(length(Xfloats),1);
path_dat(:,1) = etames;
surf_dat(:,1) = eta0;
etaemp = fft(interpft(etames,KT));
[rcnt,ccnt] = size(etaemp);
if ccnt>rcnt
    etaemp = etaemp.';
end
xf = repmat([real(etaemp(1:K));imag(etaemp(2:K));zeros(KT,1)],1,Nens) + [zeros(2*K-1,Nens);sig*randn(KT,Nens)];

% Build "true" data paths
dcnt = 2;
for kk=1:nmax
    params(5) = 14;
    un = solver_rk4(params, L1, Kmesh, mDk, tnh, eLdt, eLdt2, un);
    nvec = [un(1:K);0] + 1i*[0;un(K+1:KT-1);0];
    etan = [nvec;conj(nvec(K:-1:2))];
    
    if mod(kk,nindt) == 0
       for jj=1:Ndat
           path_dat(jj,dcnt) = -1/KT*real(sum(etan.*exp(1i*Kmesh*Xfloats(jj)))) +  sig*randn(1);
       end
        
       surf_dat(:,dcnt) = real(ifft(etan));
       dcnt = dcnt + 1;
    end
    
end

params(5) = Mval;
msqerror = zeros(nsamp);
msqerror(1) = norm(interpft(etames,KT) - eta0)/norm(eta0);

tvals = zeros(nsamp);
for jj = 2:nsamp
    rmat =  sig*randn(Ndat,Nens);
    dmat = repmat(path_dat(:,jj),1,Nens) + rmat;
    cormat = rmat*rmat';
    
    xa = analysis_step(K,Nens,xf,dmat,Hmat,cormat);
    
    xapprox = sum(xa(1:KT,:),2)/Nens;
    nvec = [xapprox(1:K);0] + 1i*[0;xapprox(K+1:KT-1);0];
    etan = [nvec;conj(nvec(K:-1:2))];
    etan(Kc:Kuc) = 0;
    msqerror(jj) = norm(real(ifft(etan))-surf_dat(:,jj))/norm(surf_dat(:,jj));
    tvals(jj) = dt*jj*nindt;
    
    if jj<nsamp
        parfor kk=1:Nens
            xf(:,kk) = solver_rk4(params,L1,Kmesh,mDk,tnh,eLdts,eLdt2s,xa(:,kk));
        end
    end
    
end        
               
xapprox = sum(xa(1:KT,:),2)/Nens;
nvec = [xapprox(1:K);0] + 1i*[0;xapprox(K+1:KT-1);0];
etan = [nvec;conj(nvec(K:-1:2))];
etan(Kc:Kuc) = 0;
fin_dat = [real(ifft(etan));surf_dat(:,nsamp)];