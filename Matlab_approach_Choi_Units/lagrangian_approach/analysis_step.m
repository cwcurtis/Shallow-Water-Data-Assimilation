function xf = analysis_step(K,Nens,xf,dmat,cormat)
    
    % xf - matrix representing ensemble of forecasts.  
    % Nens - number of members in the ensemble
    % Ndat - number of data values sampled 
    % dmat - data matrix at time t_k
    % sig - standard deviation of Gaussian distribution for errors

    KT = 2*K;

    xmean = sum(xf,2)/Nens;
    xfluc = xf - repmat(xmean,1,Nens);
    
    Pmat = xfluc*(xfluc(2*KT+1:end,:))';
    
    Pdd = Pmat(2*KT+1:end,:);
    mrhs = (Pdd + cormat)\(dmat-xf(2*KT+1:end,:));
    
    xf = xf +  Pmat*mrhs;