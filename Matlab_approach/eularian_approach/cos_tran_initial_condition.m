function eta0 = cos_tran_initial_condition(K,Llx,mu,tvals,path_dat)

Kmesh = pi/Llx*[ 0:K -K+1:-1 ]';
mDk = mu*Kmesh;
tnh = tanh(mDk);
L1 = Kmesh.*tnh/mu;
drel = sqrt(L1);
dts = tvals(2)-tvals(1);

tspan = length(tvals);

Tmat = zeros(2*K,tspan);
Tmat(:,1) = 1/2;
Tmat(:,2:tspan-1) = cos(drel*tvals(2:tspan-1));
Tmat(:,tspan) = 1/2*cos(drel*tvals(end));

eta0 = sqrt(2*K)*dts*Tmat*path_dat;
%eta0 = dts*Tmat*path_dat;
plot(-K+1:K,log10(abs(fftshift(eta0))))
pause