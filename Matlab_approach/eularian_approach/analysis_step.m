function [xf,lam] = analysis_step(K,Nens,xf,dmat,Hmat,cormat,lam,slam,sdat)
    
    % xf - matrix representing ensemble of forecasts.  
    % Nens - number of members in the ensemble
    % Ndat - number of data values sampled 
    % dmat - data matrix at time t_k
    % sig - standard deviation of Gaussian distribution for errors
    
    KT = 2*K;
    xmean = mean(xf,2);
    xfluc = xf - repmat(xmean,1,Nens);
    xfH = Hmat*xfluc(1:KT-1,:);
    
    HPf = xfH*xfH'/(Nens-1);
    Hxfm = Hmat*xmean(1:KT-1);
    y0 = mean(dmat,2);
    Dst = norm(Hxfm - y0);
    
    [Ndat,~] = size(HPf);
    spf = trace(HPf)/Ndat;
    
    b = -(sdat+lam*spf);
    c = slam*spf^2/2;
    d = -slam*spf^2*Dst^2/2;
    disc = 18*b*c*d - 4*b^3*d + b^2*c^2 - 4*c^3 - 27*d^2;
    lvals = roots([1 b c d]);     
    if disc < 0
        indsc = abs(imag(lvals)) < eps;
        lvalc = real(lvals(indsc));
    else
        lvals = real(lvals);
        [~,indc] = min(abs(lvals - lam));
        lvalc = lvals(indc);
    end
    lamp = (lvalc-sdat)/spf;
    %f = @(l) exp(-Dst^2/(2*(l*spf+sdat)))/sqrt(l*spf+sdat) * exp(-(l-lam)^2/(2*slam))/sqrt(slam);
        
    %slamp = -slam/2*log(f(lamp+sqrt(slam))/f(lamp));
    if lamp > 1
       lam = lamp;
       %slam = slamp;
       %disp('Updated inflation factor:')
       %disp(lam)
       %disp('Updated inflation variance:')
       %disp(slam)
    end
    
    xfluc = sqrt(lam)*xfluc;
    xfH = Hmat*xfluc(1:KT-1,:);
    Pf = xfluc*xfluc'/(Nens-1);
    
    mrhs = (xfH*xfH'/(Nens-1) + cormat)\(dmat-Hmat*xf(1:KT-1,:));
        
    xf = xf +  Pf(:,1:KT-1)*Hmat'*mrhs;
    