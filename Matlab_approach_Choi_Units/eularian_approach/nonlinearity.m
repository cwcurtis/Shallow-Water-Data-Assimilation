function rhs = nonlinearity(K,eta,q,G0,ep,mu,Kmesh,mDk,tnh,Mval)

KT = 2*K;
% Find the wave numbers to implement the 2/3 de-aliasing throughout
Kc = floor(2*K/3);
Kuc = KT-Kc+1;
Kc = Kc+1;

eta(Kc:Kuc)=0;
q(Kc:Kuc)=0;

etax = real(ifft(1i*Kmesh.*eta)); 
qx = real(ifft(1i*Kmesh.*q)); 

etap = real(ifft(eta));

dnohot = dno_maker(KT,etap,qx,G0,ep,mu,mDk,tnh,Mval);

rhs1 = fft(dnohot);
rhs2 = .5*ep*fft(-qx.^2 + mu^2*((G0+dnohot+ep*etax.*qx).^2)./(1+ep^2*mu^2*etax.^2));

rhs1(Kc:Kuc) = 0;
rhs2(Kc:Kuc) = 0;

rhs = [rhs1;rhs2];
