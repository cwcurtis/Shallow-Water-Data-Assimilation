function dnohot = dno_maker(KT,eta,qx,G0,ep,mu,mDk,tnh,Mval)

    % KT is number of modes used in pseudo-spectral scheme
    % eta is surface height in physical space
    % q is surface potential in physical space
    % G0 is first term of DNO in physical space
    % ep is epsilon
    % mu is mu
    % Kmesh = pi/Llx*[0:K -K+1:-1], or is essentially a derivative 
    % Mval+1 is number of terms used in DNO expansion
    
    %mDk = mu*Kmesh;
    %tnh = tanh(mDk);         
    
    phis = zeros(KT,Mval+1);    
    dnohot = zeros(KT,1);    
    phis(:,1) = G0;
    epp = 1;
    
    for jj=2:Mval+1
        phic = zeros(KT,1);
        Dkp = ones(KT,1);
        etap = ones(KT,1);        
        for ll=1:jj-2            
            Dkp = mDk.*Dkp;
            etap = eta.*etap/ll;            
            if mod(ll,2) == 0
               tvec = Dkp.*fft(etap.*phis(:,jj-ll));
            else
               tvec = Dkp.*tnh.*fft(etap.*phis(:,jj-ll));
            end                
            phic = phic + tvec;            
        end       
        Dkp = mDk.*Dkp;
        etap = eta.*etap/(jj-1);              
        if mod(jj,2) == 0
            fvec = Dkp.*(tnh.*fft(etap.*G0) + 1i/mu*fft(etap.*qx));
        else
            fvec = Dkp.*(fft(etap.*G0) + 1i/mu*tnh.*fft(etap.*qx));
        end
        phic = -real(ifft(phic + fvec));
        phis(:,jj) = phic;
        epp = epp*ep;
        dnohot = dnohot + epp*phic;                
    end        
end