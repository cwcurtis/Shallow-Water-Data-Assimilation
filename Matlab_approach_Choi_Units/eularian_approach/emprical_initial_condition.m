function [etar,etai,qr,qi] = emprical_initial_condition(K,Llx,mu,Xfloats,tvals,path_dat)

KT = 2*K+1;
Kmesh = pi/Llx*(1:K)';
mDk = mu*Kmesh;
tnh = tanh(mDk);
L1 = Kmesh.*tnh/mu;
drel = sqrt(L1);

mask = ones(K,1);
mask(1:2:K) = -1;

Nsamp = length(tvals);
Ndat = length(Xfloats);

Qmat = zeros(2*Nsamp*Ndat,4*K);
bvec = zeros(2*Nsamp*Ndat,1);
disp([2*Nsamp*Ndat 4*K])
cmat = cos(tvals'*drel');
smat = sin(tvals'*drel');

for jj=1:Ndat     
    rvec = (1+(jj-1)*Nsamp:jj*Nsamp);
    Nvec = Nsamp + rvec;
    
    bvec(rvec) = path_dat(jj,:);
    bvec(Nvec) = path_dat(jj+Ndat,:);
    
    lmatc = diag(mask.*cos(Kmesh*Xfloats(jj)));
    lmats = diag(mask.*sin(Kmesh*Xfloats(jj)));
    
    q11 = cmat*lmatc;
    q12 = -cmat*lmats;
    q13 = smat*lmatc;
    q14 = -smat*lmats;
    
    q21 = -q13;
    q22 = -q14;
    q23 = q11;
    q24 = q12;
    
    Qmat(rvec,1:K) = q11;
    Qmat(rvec,K+1:2*K) = q12;
    Qmat(rvec,2*K+1:3*K) = q13;
    Qmat(rvec,3*K+1:4*K) = q14;
    
    Qmat(Nvec,1:K) = q21;
    Qmat(Nvec,K+1:2*K) = q22;
    Qmat(Nvec,2*K+1:3*K) = q23;
    Qmat(Nvec,3*K+1:4*K) = q24;
    
end

xsol = KT/2*Qmat\bvec;
bemp = 2/KT*Qmat*xsol;

disp(norm(bvec - bemp,'inf'))

plot(tvals,path_dat(1,:),'--',tvals,bemp(1:Nsamp),'-')
pause

etar = xsol(1:K);
etai = xsol(K+1:2*K);

qr = xsol(2*K+1:3*K);
qi = xsol(3*K+1:4*K);