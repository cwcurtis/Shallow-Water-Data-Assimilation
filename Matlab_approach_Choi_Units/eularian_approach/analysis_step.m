function xf = analysis_step(K,Nens,xf,dmat,Hmat,cormat)
    
    % xf - matrix representing ensemble of forecasts.  
    % Nens - number of members in the ensemble
    % Ndat - number of data values sampled 
    % dmat - data matrix at time t_k
    % sig - standard deviation of Gaussian distribution for errors
    KT = 2*K;
    xmean = sum(xf,2)/Nens;
    xfluc = xf - repmat(xmean,1,Nens);
    
    xfH = Hmat*xfluc(1:KT-1,:);
    mrhs = (xfH*xfH' + cormat)\(dmat-Hmat*xf(1:KT-1,:));
        
    xf = xf +  xfluc*xfH'*mrhs;