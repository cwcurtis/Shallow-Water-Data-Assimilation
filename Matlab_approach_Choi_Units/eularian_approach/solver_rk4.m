function  usol = solver_rk4(params, L1, Kmesh, mDk, tnh, eLdt, eLdt2, uint)
    
    K= params(1); ep=params(2); mu=params(3); Mval=params(5); dt = params(6);
        
    KT = 2*K;
    Kc = floor(KT/3);
    Kuc = KT - Kc + 1;
    Kc = Kc + 1;
    
    nvec = [uint(1:K);0] + 1i*[0;uint(K+1:KT-1);0];
    etan = [nvec;conj(nvec(K:-1:2))];
    qn = fft(uint(KT:2*KT-1));
    
    ufreq = [etan;qn];
    
    G0 = real(ifft(L1.*qn));
    k1 = dt*nonlinearity(K,etan,qn,G0,ep,mu,Kmesh,mDk,tnh,Mval); 
    
    svec = eLdt2*(ufreq + .5*k1);

    etah2 = svec(1:KT);
    qh2 = svec(KT+1:2*KT);
    etah2(Kc:Kuc) = 0;
    qh2(Kc:Kuc) = 0;

    G0 = real(ifft(L1.*qh2));
    k2 = dt*nonlinearity(K,etah2,qh2,G0,ep,mu,Kmesh,mDk,tnh,Mval);
    
    svec = eLdt2*ufreq + .5*k2;

    etah2 = svec(1:KT);
    qh2 = svec(KT+1:2*KT);
    etah2(Kc:Kuc) = 0;
    qh2(Kc:Kuc) = 0;

    G0 = real(ifft(L1.*qh2));
    k3 = dt*nonlinearity(K,etah2,qh2,G0,ep,mu,Kmesh,mDk,tnh,Mval);
    
    svec = eLdt*ufreq + eLdt2*k3;

    etah2 = svec(1:KT);
    qh2 = svec(KT+1:2*KT);
    etah2(Kc:Kuc) = 0;
    qh2(Kc:Kuc) = 0;

    G0 = real(ifft(L1.*qh2));
    k4 = dt*nonlinearity(K,etah2,qh2,G0,ep,mu,Kmesh,mDk,tnh,Mval);
    
    ufreq = eLdt*(ufreq+k1/6)+ eLdt2*(k2+k3)/3 + k4/6;

    etan = ufreq(1:KT);
    qn = ufreq(KT+1:2*KT);

    etan(Kc:Kuc) = 0;
    qn(Kc:Kuc) = 0;
    
    usol = [real(etan(1:K));imag(etan(2:K));real(ifft(qn))];