function [etap,etan,qp,qn] = emprical_initial_condition_alt(K,Llx,mu,Xfloats,tvals,path_dat)

KT = 2*K+1;
Kmesh = pi/Llx*(1:K)';
mDk = mu*Kmesh;
tnh = tanh(mDk);
L1 = Kmesh.*tnh/mu;
drel = sqrt(L1);
dmat = diag(drel);
dimat = diag(drel.^(-1));

mask = ones(K,1);
mask(1:2:K) = -1;

Nsamp = length(tvals);
Ndat = length(Xfloats);

cmat = cos(tvals'*drel');
smat = sin(tvals'*drel');
s12 = smat*dmat;
s21 = -smat*dimat;
Qmat = [cmat s12; s21 cmat];

realndat = zeros(Nsamp*Ndat,1);
realqdat = zeros(Nsamp*Ndat,1);

for jj=1:Ndat     
    rvec = (1+(jj-1)*Nsamp:jj*Nsamp);
    Nvec = Nsamp + rvec;
    bn = path_dat(jj,:);
    bq = path_dat(jj+Ndat,:);
    tvec = Qmat\[bn';bq'];
    realndat(rvec) = KT*tvec(1:Nsamp);
    realqdat(Nvec) = KT*tvec(Nsamp+1:2*Nsamp);
end

msk = -.5*1i*mask./sin(Kmesh.*(Xfloats(1)-Xfloats(2)));
eval1 = exp(-1i*Kmesh*Xfloats(1));
eval2 = exp(-1i*Kmesh*Xfloats(2));

etap = msk.*(eval2.*realndat(1:Nsamp) - eval1.*realndat(Nsamp+1:2*Nsamp));
etan = msk.*(-conj(eval2).*realndat(1:Nsamp) + conj(eval1).*realndat(Nsamp+1:2*Nsamp));

qp = msk.*(eval2.*realqdat(1:Nsamp) - eval1.*realqdat(Nsamp+1:2*Nsamp));
qn = msk.*(-conj(eval2).*realqdat(1:Nsamp) + conj(eval1).*realqdat(Nsamp+1:2*Nsamp));