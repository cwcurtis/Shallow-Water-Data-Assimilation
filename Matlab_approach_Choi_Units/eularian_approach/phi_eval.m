function rhs = phi_eval(params, xvec, eta, q, Kmesh, mDk, tnh)

    K = params(1); ep = params(2); mu = params(3); Mval = params(5);

    KT = 2*K;
    xval = xvec(1);
    zval = xvec(2);
    phis = zeros(KT,Mval+1);
    
    evec = exp(1i*Kmesh*xval);
    ovec = ones(KT,1);
    tnvec = tanh(mDk*zval);
    cvec = cosh(mDk*zval);
    etap = real(ifft(eta));

    phis(:,1) = q;
    phif = q;
    eppow = ep;
       
    for jj = 2:Mval+1
       
        etapow = ovec;
        Dkpow = ovec;
        phic = zeros(KT,1);
        
        for ll = 1:jj-1         
            
            Dkpow = Dkpow.*Kmesh;
            
            if mod(ll,2) == 0
                tvec = real(ifft(Dkpow.*phis(:,jj-ll)));
            else
                tvec = real(ifft(tnh.*Dkpow.*phis(:,jj-ll)));
            end            
            
            etapow = mu*etapow.*etap/ll;
            phic = phic + etapow.*tvec;
            
        end
        
        phis(:,jj) = -fft(phic);
        phif = phif + eppow*phis(:,jj);
        eppow = eppow*ep;
        
    end
    
    phif = phif.*cvec;
    
    xnew = -ep/KT*real(sum(1i*Kmesh.*phif.*(ovec + tnvec.*tnh).*evec));
    znew = -ep/KT*real(sum(mDk/mu.*phif.*(tnvec/mu + tnh/mu).*evec));
    
    rhs = [xnew;znew];
    
end

