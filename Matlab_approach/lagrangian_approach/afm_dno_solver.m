function [etaf,etat] = afm_dno_solver(K,ep,mu,Llx,tf,Mval,dt)

    KT = 2*K;
    % Find the wave numbers to implement the 2/3 de-aliasing throughout
    Kc = floor(2*K/3);
    Kuc = KT-Kc+1;
    Kc = Kc+1;
    
    Xmesh = linspace(-Llx,Llx,KT+1);
    Xmesh = Xmesh(1:KT)';    
    
    Kmesh = pi/Llx*[0:K -K+1:-1]';
        
    nmax = round(tf/dt);
    
    L1 = Kmesh.*tanh(mu.*Kmesh)/mu;
    
    Linvd = (1 + 9*dt^2/16*L1).^(-1);
    Linv12 = 3*dt/4*L1.*Linvd;
    Linv21 = -3*dt/4*Linvd;
    
    etan = fft(cos(pi*Xmesh/Llx));
    qn = fft(sin(pi*Xmesh/Llx));
    
    etan(Kc:Kuc) = 0;
    qn(Kc:Kuc) = 0;
    G0 = real(ifft(L1.*qn));
    
    etanm1 = etan;
    qnm1 = qn;
    
    nln = nonlinearity(K,etan,qn,G0,ep,mu,Llx,Kmesh,Mval);
    etat = real(ifft(nln(1:KT)));
    
    nlnm1 = nln;
    nlnm2 = nlnm1;
    nlnm3 = nlnm2;
    
    for jj=1:nmax
        
        G0 = real(ifft(L1.*qn));
        nln = nonlinearity(K,etan,qn,G0,ep,mu,Llx,Kmesh,Mval);
        
        nlvecn = 55/24*nln(1:KT) - 59/24*nlnm1(1:KT) + 37/24*nlnm2(1:KT) - 3/8*nlnm3(1:KT);    
        nlvecq = 55/24*nln(KT+1:2*KT) - 59/24*nlnm1(KT+1:2*KT) + 37/24*nlnm2(KT+1:2*KT) - 3/8*nlnm3(KT+1:2*KT);
        
        eta1 = Linvd.*(etan + 1/3*etanm1 + dt*nlvecn);
        eta2 = Linv12.*(qn + 1/3*qnm1 + dt*nlvecq);
        
        q1 = Linvd.*(qn + 1/3*qnm1 + dt*nlvecq);
        q2 = Linv21.*(etan + 1/3*etanm1 + dt*nlvecn);
        
        etanp1 = -etanm1/3 + eta1 + eta2;
        qnp1 = -qnm1/3 + q1 + q2;        
        
        etanm1 = etan;
        etan = etanp1;
        
        qnm1 = qn;
        qn = qnp1;
        
        nlnm3 = nlnm2;
        nlnm2 = nlnm1;
        nlnm1 = nln;
        
    end
    
    etaf = real(ifft(etan));
    %plot(Xmesh,etaf)