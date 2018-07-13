function xf = analysis_step(K,Nens,xf,dmat,Hmat,cormat)
    
    % xf - matrix representing ensemble of forecasts.  
    % Nens - number of members in the ensemble
    % Ndat - number of data values sampled 
    % dmat - data matrix at time t_k
    % sig - standard deviation of Gaussian distribution for errors
    KT = 2*K;
    xmean = mean(xf,2);
    xfluc = xf - repmat(xmean,1,Nens);
    xfH = Hmat*xfluc(1:KT-1,:);
        
    Pf = xfluc*xfluc'/(Nens-1);
    
    disp(num2str(xfH*xfH'/(Nens-1) + cormat))
    pause
    
    mrhs = (xfH*xfH'/(Nens-1) + cormat)\(dmat-Hmat*xf(1:KT-1,:));
        
    xf = xf +  Pf(:,1:KT-1)*Hmat'*mrhs;