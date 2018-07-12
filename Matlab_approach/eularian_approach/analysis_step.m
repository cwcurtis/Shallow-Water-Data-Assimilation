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
    [Pf1,~] = size(xfluc);
    [HPf1,~] = size(xfH);
    Pf = zeros(Pf1);
    HPf = zeros(HPf1);
    
    parfor ll=1:Nens
        tvec = xfluc(:,ll);
        htvec = xfH(:,ll);
        Pf = Pf + tvec*tvec';
        HPf = HPf + htvec*htvec'; 
    end
    Pf = Pf/(Nens-1);
    HPf = HPf/(Nens-1);
    
    mrhs = (HPf*HPf' + cormat)\(dmat-Hmat*xf(1:KT-1,:));
        
    xf = xf +  Pf(:,1:KT-1)*Hmat'*mrhs;