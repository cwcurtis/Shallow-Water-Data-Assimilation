function rhs = phi_eval_surf(params, xvec, eta, q, Kmesh, dno)

    K = params(1); ep = params(2); mu = params(3);

    KT = 2*K;
    xval = xvec(1);
    evec = exp(1i*Kmesh*xval);
    
    qx = -1/KT*real(sum( 1i*Kmesh.*q.*evec ));
    etax = -1/KT*real(sum( 1i*Kmesh.*eta.*evec ));
    etat = -1/KT*real(sum( dno.*evec ));
    xdot = (qx-ep*mu^2*etax*etat)/(1 + (ep*mu)^2*etax^2);
    zdot = (etat + ep*etax*qx)/(1 + (ep*mu)^2*etax^2);
    
    rhs = [ep*xdot;ep*zdot];
    
end

