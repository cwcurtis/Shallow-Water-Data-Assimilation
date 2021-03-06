function [smth_dat,path_dat,surf_dat,tvals,msqerror] = kalman_filter_temporal(K,Llx,tf,dt,dts,Nens,sig,Xfloats)

KT = 2*K;
Kc = floor(KT/3);
Kuc = KT - Kc + 1;
Kc = Kc + 1;
dx = Llx/K;    
ep = .1;
mu = sqrt(ep);
Mval = 2;
%Xmesh = linspace(-Llx,Llx-dx,KT);
Xmesh = -Llx:dx:Llx-dx;

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
Hmat = 1/KT*Hmat;

Kmesh = pi/Llx*[ 0:K -K+1:-1 ]';
Kvals = -K+1:K;
mDk = mu*Kmesh;
tnh = tanh(mDk);
L1 = Kmesh.*tnh/mu;
drel = sqrt(L1);

k0 = pi/Llx;
rvec = exp(-(Kmesh(1:K+1)-k0).^2/(2*sqrt(sig)));
rvec(1) = 0;
rvec(K+1) = 0;
asub = rvec.*exp(2*pi*1i*rand(K+1,1));
avals = [asub;conj(asub(K:-1:2))];
avals = sqrt(KT)*avals/norm(avals);

bsub = rvec.*exp(2*pi*1i*rand(K+1,1));
bvals = [bsub;conj(bsub(K:-1:2))];
bvals = sqrt(KT)*bvals/norm(bvals);

eta0 = real(ifft(avals));
q0 = real(ifft(bvals));
etaf = avals;
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

freqn = [avals(K+2:KT);avals(1:K+1)];
etamesno = (1/KT)*real(sum(repmat(freqn.',Ndat,1).*exp(1i*pi/Llx*(Xfloats+Llx)*Kvals),2));

scfac = max(abs(etamesno));
etames = etamesno + sig*scfac*randn(Ndat,1);
msqerror = zeros(nsamp);
msqerror(1) = sqrt(Llx/K)*norm(etames - etamesno);

path_dat(1:Ndat,1) = etames;
surf_dat(1:Ndat,1) = etamesno;
smth_dat(1:Ndat,1) = etamesno;

% Build "true" and "sampled" data paths
dcnt = 2;
for kk=1:nmax-1
    params(5) = 14;
    un = solver_rk4(params, L1, Kmesh, mDk, tnh, eLdt, eLdt2, un);
    nvec = [un(1:K);0] + 1i*[0;un(K+1:KT-1);0];
    etan = [nvec;conj(nvec(K:-1:2))];    
    if mod(kk,nindt) == 0
       freqn = [etan(K+2:KT);etan(1:K+1)];
       tmpval = 1/KT*real(sum(repmat(freqn.',Ndat,1).*exp(1i*pi/Llx*(Xfloats+Llx)*Kvals),2));
       surf_dat(:,dcnt) = tmpval;
       path_dat(:,dcnt) = tmpval + scfac*sig*randn(Ndat,1);
       tvals(dcnt) = dt*(dcnt-1)*nindt;
       dcnt = dcnt + 1;
    end    
end

%etaemp = fft(interpft(etames,KT));
etamesfreq = cos_tran_initial_condition(K,Llx,mu,tvals,path_dat(1,:)');
etames = real(ifft(etamesfreq));
figure(1)
plot(Xmesh,log10(abs(fftshift(sqrt(KT)*fft(eta0)))),'--',Xmesh,log10(abs(fftshift(etamesfreq))),'-')
pause

etamesnxt = path_dat(:,2);
etaempnxt = fft(interpft(etamesnxt,KT));
q0freq = (etaempnxt - cos(dts*drel).*etaemp)./(drel.*sin(dts*drel));
q0freq(1) = 0;
q0emp = real(ifft(q0freq));

%xf = repmat([real(etaemp(1:K));imag(etaemp(2:K));q0emp],1,Nens);
xf = repmat([real(etaemp(1:K));imag(etaemp(2:K));q0],1,Nens);
figure(2)
plot(Xmesh,q0,'--',Xmesh,q0emp,'-')
params(5) = Mval;

for jj = 2:nsamp
    rmat =  scfac*sig*randn(Ndat,Nens);
    dmat = repmat(path_dat(:,jj),1,Nens) + rmat;
    cormat = rmat*rmat';
    
    xa = analysis_step(K,Nens,xf,dmat,Hmat,cormat);
    
    napprox = sum(xa(1:KT-1,:),2)/Nens;
    nvec = [napprox(1:K);0] + 1i*[0;napprox(K+1:KT-1);0];
    etan = [nvec;conj(nvec(K:-1:2))];
    etan(Kc:Kuc) = 0;
    freqn = [etan(K+2:KT);etan(1:K+1)];
    smth_dat(:,jj) = 1/KT*real(sum(repmat(freqn.',Ndat,1).*exp(1i*pi/Llx*(Xfloats+Llx)*Kvals),2));
            
    msqerror(jj) = sqrt(Llx/K)*norm(smth_dat(:,jj)-surf_dat(:,jj));
    
    if jj<nsamp
        parfor kk=1:Nens
            xf(:,kk) = solver_rk4(params,L1,Kmesh,mDk,tnh,eLdts,eLdt2s,xa(:,kk));
        end
    end    
end        