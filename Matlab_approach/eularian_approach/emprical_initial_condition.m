function [etar,etai,qem0] = emprical_initial_condition(K,Llx,mu,Xfloats,tvals,path_dat)

KT = 2*K;
Ksc = pi*(K+1)/Llx;
Kmesh = pi/Llx*(1:K);
mDk = mu*Kmesh;
tnh = tanh(mDk);
L1 = Kmesh.*tnh/mu;
drel = sqrt(L1);

mask = ones(1,K);
mask(1:2:K) = -1;

Nsamp = length(tvals);
Ndat = length(Xfloats);

Qmat = zeros(Nsamp*Ndat,4*K);
bvec = zeros(Nsamp*Ndat,1);

cmat = cos(tvals'*drel);
smat = sin(tvals'*drel);

for jj=1:Ndat 
    
    csmat = repmat(mask.*cos(Xfloats(jj).*(Kmesh-Ksc)).*exp(1i*Ksc*Xfloats(jj)),Nsamp,1);
    snmat = repmat(mask.*sin(Xfloats(jj).*(Kmesh-Ksc)).*exp(1i*Ksc*Xfloats(jj)),Nsamp,1);
    
    rvec = 1+(jj-1)*Nsamp:jj*Nsamp;
    bvec(rvec) = path_dat(jj,:);
    
    q11 = 2/KT*csmat.*cmat;
    q12 = -2/KT*snmat.*cmat;
    q21 = 2/KT*csmat.*smat;
    q22 = -2/KT*snmat.*smat;
    
    Qmat(rvec,1:K) = q11;
    Qmat(rvec,K+1:2*K) = q12;
    Qmat(rvec,2*K+1:3*K) = q21;
    Qmat(rvec,3*K+1:4*K) = q22;
    
end

xsol = real(bvec\Qmat);
disp(size(xsol))
%{
[U,S,V] = svd(Qmat);
Svec = diag(S);
Vmat = V';
Sinds = Svec > 1e-4;
bvec = U'*bvec;
xsol = bvec(Sinds)\Vmat(Sinds,:);
disp(Svec(Sinds))
disp([length(xsol) K])
pause
%}

etar = xsol(1:K).';
etai = xsol(K+1:2*K).';
qr = xsol(2*K+1:3*K).';
qi = xsol(3*K+1:4*K).';

qem0 = qr + 1i*qi;