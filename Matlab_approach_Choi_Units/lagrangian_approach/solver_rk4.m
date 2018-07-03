function  usol = solver_rk4(params, L1, Kmesh, mDk, tnh, eLdt, eLdt2, uint, Ndat)
    
    K= params(1); ep=params(2); mu=params(3); Mval=params(5); dt = params(6);
        
    KT = 2*K;
    Kc = floor(KT/3);
    Kuc = KT - Kc + 1;
    Kc = Kc + 1;
    
    k1t = zeros(2*Ndat,1);
    k2t = zeros(2*Ndat,1);
    k3t = zeros(2*Ndat,1);
    k4t = zeros(2*Ndat,1);
    
    etan = fft(uint(1:KT));
    qn = fft(uint(KT+1:2*KT));
    xvi = uint(2*KT+1:end);
    
    etan(Kc:Kuc) = 0;
    qn(Kc:Kuc) = 0;
    ufreq  = [etan;qn];
    
    %disp(etan)
    
    G0 = real(ifft(L1.*qn));
    [nl,dno] = nonlinearity(K,etan,qn,G0,ep,mu,Kmesh,mDk,tnh,Mval); 
    k1 = dt*nl;

    for jj =1:Ndat
        %k1t(2*jj-1:2*jj) = dt*phi_eval(params,xvi(2*jj-1:2*jj),etan,qn,Kmesh,mDk,tnh);
        k1t(2*jj-1:2*jj) = dt*phi_eval_surf(params,xvi(2*jj-1:2*jj),etan,qn,Kmesh,dno);
    end
                                       
    svec = eLdt2*(ufreq + .5*k1);

    etah2 = svec(1:KT);
    qh2 = svec(KT+1:2*KT);
    etah2(Kc:Kuc) = 0;
    qh2(Kc:Kuc) = 0;

    G0 = real(ifft(L1.*qh2));
    [nl,dno] = nonlinearity(K,etah2,qh2,G0,ep,mu,Kmesh,mDk,tnh,Mval);
    k2 = dt*nl;

    for jj = 1:Ndat
        %k2t(2*jj-1:2*jj) = dt*phi_eval(params,xvi(2*jj-1:2*jj)+k1t(2*jj-1:2*jj)/2,etah2,qh2,Kmesh,mDk,tnh);
        k2t(2*jj-1:2*jj) = dt*phi_eval_surf(params,xvi(2*jj-1:2*jj)+k1t(2*jj-1:2*jj)/2,etah2,qh2,Kmesh,dno);
    end
    
    svec = eLdt2*ufreq + .5*k2;

    etah2 = svec(1:KT);
    qh2 = svec(KT+1:2*KT);
    etah2(Kc:Kuc) = 0;
    qh2(Kc:Kuc) = 0;

    G0 = real(ifft(L1.*qh2));
    [nl,dno] = nonlinearity(K,etah2,qh2,G0,ep,mu,Kmesh,mDk,tnh,Mval);
    k3 = dt*nl;

    for jj = 1:Ndat
        %k3t(2*jj-1:2*jj) = dt*phi_eval(params,xvi(2*jj-1:2*jj)+k2t(2*jj-1:2*jj)/2,etah2,qh2,Kmesh,mDk,tnh);
        k3t(2*jj-1:2*jj) = dt*phi_eval_surf(params,xvi(2*jj-1:2*jj)+k2t(2*jj-1:2*jj)/2,etah2,qh2,Kmesh,dno);
    end
    
    svec = eLdt*ufreq + eLdt2*k3;

    etah2 = svec(1:KT);
    qh2 = svec(KT+1:2*KT);
    etah2(Kc:Kuc) = 0;
    qh2(Kc:Kuc) = 0;

    G0 = real(ifft(L1.*qh2));
    [nl,dno] = nonlinearity(K,etah2,qh2,G0,ep,mu,Kmesh,mDk,tnh,Mval);
    k4 = dt*nl;

    for jj = 1:Ndat
        %k4t(2*jj-1:2*jj) = dt*phi_eval(params,xvi(2*jj-1:2*jj)+k3t(2*jj-1:2*jj),etah2,qh2,Kmesh,mDk,tnh);
        k4t(2*jj-1:2*jj) = dt*phi_eval_surf(params,xvi(2*jj-1:2*jj)+k3t(2*jj-1:2*jj),etah2,qh2,Kmesh,dno);
    end
    
    ufreq = eLdt*(ufreq+k1/6)+ eLdt2*(k2+k3)/3 + k4/6;

    xfv = xvi+(k1t + 2*(k2t+k3t) + k4t)/6;

    etan = ufreq(1:KT);
    qn = ufreq(KT+1:2*KT);

    etan(Kc:Kuc) = 0;
    qn(Kc:Kuc) = 0;
    
    etan = real(ifft(etan));
    qn = real(ifft(qn));
    usol = [etan;qn;xfv];