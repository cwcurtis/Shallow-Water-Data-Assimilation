function [smth_dat,path_dat,surf_dat,tvals,msqerror] = kalman_filter_temporal(K,Llx,tf,dt,dts,Nens,sig,Xfloats,avals,bvals)

KT = 2*K;
Kc = floor(KT/3);
Kuc = KT - Kc + 1;
Kc = Kc + 1;
dx = Llx/K;    
ep = .1;
mu = sqrt(ep);
Mval = 2;
Xmesh = linspace(-Llx,Llx-dx,KT);

params = [K ep mu Llx Mval dt dts Nens sig];
mask = ones(KT,1);
mask(2:2:KT) = -1;

Ndat = length(Xfloats);
nindt = round(dts/dt);
nmax = round(tf/dt);
nsamp = round(nmax/nindt);

Hmat = ones(Ndat,KT-1);
evec = exp(1i*pi*Xfloats/Llx);

for jj=2:K   
    Hmat(:,jj) = Hmat(:,jj-1).*evec;
end

Hmat(:,2:K) = repmat(mask(2:K)',Ndat,1).*Hmat(:,2:K);
tmat = -2*imag(Hmat(:,2:K));
Hmat(:,2:K) = 2*real(Hmat(:,2:K));
Hmat(:,K+1:KT-1) = tmat;
Hmat = 1/KT*Hmat;

Kmesh = pi/Llx*[ 0:K -K+1:-1 ]';
mDk = mu*Kmesh;
tnh = tanh(mDk);
L1 = Kmesh.*tnh/mu;

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
surf_dat = zeros(Ndat,nsamp);
smth_dat = zeros(Ndat,nsamp);
tvals = zeros(1,nsamp);

% Build initial ensemble of forecasts
% First, use what interpolatory data we have.   

etamesno = zeros(length(Xfloats),1);
for jj=1:Ndat
   etamesno(jj) =  1/KT*real(sum(mask.*avals.*exp(1i*Kmesh*Xfloats(jj))));
end

scfac = max(abs(etamesno));
etames = etamesno + sig*scfac*randn(Ndat,1);
msqerror = zeros(nsamp);
msqerror(1) = sqrt(Llx/K)*norm(etames - etamesno);

path_dat(:,1) = etames;
surf_dat(:,1) = etamesno;
smth_dat(:,1) = etamesno;

% Build "true" and "sampled" data paths
dcnt = 2;
for kk=1:nmax-1
    params(5) = 14;
    un = solver_rk4(params, L1, Kmesh, mDk, tnh, eLdt, eLdt2, un);
    nvec = [un(1:K);0] + 1i*[0;un(K+1:KT-1);0];
    etan = [nvec;conj(nvec(K:-1:2))];    
    if mod(kk,nindt) == 0
       for jj=1:Ndat
           tmpval = 1/KT*real(sum(mask.*etan.*exp(1i*Kmesh*Xfloats(jj))));
           surf_dat(jj,dcnt) = tmpval;
           path_dat(jj,dcnt) = tmpval + scfac*sig*randn(1);           
       end      
       tvals(dcnt) = dt*(dcnt-1)*nindt;
       dcnt = dcnt + 1;
    end    
end

Ksamp = floor(nsamp/4);

[etaem0r,etaem0i,qem0] = emprical_initial_condition(Ksamp,Llx,mu,Xfloats,tvals,path_dat);

n0subfr = [0;etaem0r;zeros(K-Ksamp,1)];
n0subfi = [0;etaem0i;zeros(K-Ksamp,1)];
n0subf = n0subfr + 1i*n0subfi;

n0em = real(ifft([n0subf;conj(n0subf(K:-1:2))]));

q0subf = [0;qem0;zeros(K-Ksamp,1)];
q0emp = real(ifft([q0subf;conj(q0subf(K:-1:2))]));

plot(Xmesh,n0em,'-',Xmesh,eta0,'--')
pause

xf = repmat([n0subfr(1:K);n0subfi(2:K);q0emp],1,Nens);
params(5) = Mval;

for jj = 2:nsamp
    rmat =  scfac*sig*randn(Ndat,Nens);
    dmat = repmat(path_dat(:,jj),1,Nens) + rmat;
    cormat = rmat*rmat';
    
    xa = analysis_step(K,Nens,xf,dmat,Hmat,cormat);
    
    xapprox = sum(xa(1:KT,:),2)/Nens;
    nvec = [xapprox(1:K);0] + 1i*[0;xapprox(K+1:KT-1);0];
    etan = [nvec;conj(nvec(K:-1:2))];
    etan(Kc:Kuc) = 0;
    
    for ll=1:Ndat
        smth_dat(ll,jj) =  1/KT*real(sum(mask.*etan.*exp(1i*Kmesh*Xfloats(ll))));
    end
    
    msqerror(jj) = sqrt(Llx/K)*norm(smth_dat(:,jj)-surf_dat(:,jj));
    
    if jj<nsamp
        parfor kk=1:Nens
            xf(:,kk) = solver_rk4(params,L1,Kmesh,mDk,tnh,eLdts,eLdt2s,xa(:,kk));
        end
    end    
end        