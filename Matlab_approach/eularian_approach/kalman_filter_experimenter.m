function [fin_dat,tvals,msqerror] = kalman_filter_experimenter(K,Llx,tf,dt,dts,Nens,sig,Xfloats,avals,bvals)

KT = 2*K;
Kc = floor(KT/3);
Kuc = KT - Kc + 1;
Kc = Kc + 1;
    
ep = .1;
mu = sqrt(ep);
Mval = 3;  % number of terms in the DNO expansion

params = [K ep mu Llx Mval dt dts Nens sig];
Ndat = length(Xfloats);
nindt = round(dts/dt);
nmax = round(tf/dt);
nsamp = round(nmax/nindt);

mask = ones(KT,1);
mask(2:2:K) = -1;
mask(K+2:2:KT) = -1;

smask = ones(1,K-1);
smask(1:2:K-1)=-1;
maskm = repmat(smask,Ndat,1);

Hmat = ones(Ndat,KT-1);
evec = exp(1i*pi*Xfloats/Llx);

for jj=2:K   
    Hmat(:,jj) = Hmat(:,jj-1).*evec;
end

tmat = -2*imag(Hmat(:,2:K));
Hmat(:,2:K) = 2*real(Hmat(:,2:K)).*maskm;
Hmat(:,K+1:KT-1) = tmat.*maskm;
Hmat = 1/KT*Hmat;

Kmesh = pi/Llx*[ 0:K -K+1:-1 ]';
mDk = mu*Kmesh;
tnh = tanh(mDk);
L1 = Kmesh.*tnh/mu;
drel = sqrt(L1);

eta0 = real(ifft(avals));
q0 = real(ifft(bvals));

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
surf_dat(:,1) = eta0;

% Build "true" and "sampled" data paths
%
dcnt = 2;
for kk=1:nmax
    params(5) = 14;
    un = solver_rk4(params, L1, Kmesh, mDk, tnh, eLdt, eLdt2, un);
    nvec = [un(1:K);0] + 1i*[0;un(K+1:KT-1);0];
    etan = [nvec;conj(nvec(K:-1:2))];
    
    if mod(kk,nindt) == 0
       for jj=1:Ndat
           path_dat(jj,dcnt) = 1/KT*real(sum(mask.*etan.*exp(1i*Kmesh*Xfloats(jj)))) +  sig*randn(1);
       end
        
       surf_dat(:,dcnt) = real(ifft(etan));
       dcnt = dcnt + 1;
    end
    
end

% Build initial ensemble of forecasts
% First, use what interpolatory data we have.   

k0 = pi/Llx;
width = 1e-2;
rvec = exp(-(Kmesh-k0).^2/(2*width));
rvec(1) = 0;
xf = zeros(2*K-1+KT,Nens);

parfor jj=1:Nens
    pveca = exp(2*pi*1i*rand(K-1,1)); 
    pvecb = exp(2*pi*1i*rand(K-1,1));
    
    avals = KT*rvec/norm(rvec,2).*[0;pveca;0;conj(flipud(pveca))];
    bvals = KT*rvec/norm(rvec,2).*[0;pvecb;0;conj(flipud(pvecb))];
    q0emp = real(ifft(bvals));
    xf(:,jj) = [real(avals(1:K));imag(avals(2:K));q0emp];
end

params(5) = Mval;
msqerror = zeros(nsamp,1);
nemp = mean(xf(1:KT-1,:),2);
nvec = [nemp(1:K);0] + 1i*[0;nemp(K+1:KT-1);0];
etames = real(ifft([nvec;conj(nvec(K:-1:2))]));
    
msqerror(1) = sqrt(Llx/K)*norm(etames - surf_dat(:,1));
%path_dat(:,1) = etames;

tvals = zeros(nsamp,1);
for jj = 2:nsamp
    rmat =  10*sig*randn(Ndat,Nens);
    dmat = repmat(path_dat(:,jj),1,Nens) + rmat;
    cormat = zeros(Ndat);
    parfor ll=1:Nens
       cormat = cormat + rmat(:,ll)*rmat(:,ll)'; 
    end
    cormat = cormat/(Nens-1);
    
    xa = analysis_step(K,Nens,xf,dmat,Hmat,cormat);
    
    xapprox = mean(xa(1:KT-1,:),2);
    nvec = [xapprox(1:K);0] + 1i*[0;xapprox(K+1:KT-1);0];
    etan = [nvec;conj(nvec(K:-1:2))];
    etan(Kc:Kuc) = 0;
    msqerror(jj) = sqrt(Llx/K)*norm(real(ifft(etan))-surf_dat(:,jj));
    tvals(jj) = dt*jj*nindt;
    
    if jj<nsamp
        parfor kk=1:Nens
            xf(:,kk) = solver_rk4(params,L1,Kmesh,mDk,tnh,eLdts,eLdt2s,xa(:,kk));
        end
    end
    
end        
               
xapprox = mean(xa(1:KT-1,:),2);
nvec = [xapprox(1:K);0] + 1i*[0;xapprox(K+1:KT-1);0];
etan = [nvec;conj(nvec(K:-1:2))];
etan(Kc:Kuc) = 0;
fin_dat = [real(ifft(etan));surf_dat(:,nsamp)];