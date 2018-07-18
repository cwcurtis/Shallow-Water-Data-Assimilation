function data_plotter(K,Llx,tf,dt,dts,Nens,sig,Nplates)

KT = 2*K;

Xfloats = linspace(-Llx,Llx,Nplates/2)';
Kmesh = pi/Llx*[ 0:K -K+1:-1 ]';
k0 = pi/Llx;

width = 1e-1;
rvec = exp(-(Kmesh-k0).^2/(2*width^2));
rvec(1) = 0;
arand = 2*pi*1i*rand(K-1,1);
brand = 2*pi*1i*rand(K-1,1);
avals = KT*rvec/norm(rvec,2).*exp([0;arand;0;conj(flipud(arand))]);
bvals = KT*rvec/norm(rvec,2).*exp([0;brand;0;conj(flipud(brand))]);

[fin_dat_sfloat,tvals,msqerror_sfloat] = kalman_filter_experimenter(K,Llx,tf,dt,dts,Nens,sig,width,k0,Xfloats,avals,bvals);

Xfloats = linspace(-Llx,Llx,Nplates)';
[fin_dat_tfloat,tvals,msqerror_tfloat] = kalman_filter_experimenter(K,Llx,tf,dt,dts,Nens,sig,width,k0,Xfloats,avals,bvals);

approx_sf = fin_dat_sfloat(1:KT);
exact_sol = fin_dat_sfloat(KT+1:2*KT);
approx_tf = fin_dat_tfloat(1:KT);

dx = Llx/K;
Xvals = (-Llx:dx:Llx-dx)';

sf_error = log10(abs(approx_sf - exact_sol));
tf_error = log10(abs(approx_tf - exact_sol));

Nbin = 80;
eraxis = linspace(-5,.01,Nbin);
sf_dist = zeros(Nbin,1);
tf_dist = zeros(Nbin,1);

for jj=1:Nbin-1
   
    for ll=1:KT
       if eraxis(jj)<= sf_error(ll) && sf_error(ll)<eraxis(jj+1) 
            sf_dist(jj) = sf_dist(jj) + 1;
       end
       if eraxis(jj)<= tf_error(ll) && tf_error(ll)<eraxis(jj+1) 
            tf_dist(jj) = tf_dist(jj) + 1;
       end
    end
    
end

sf_dist = sf_dist/sum(sf_dist);
tf_dist = tf_dist/sum(tf_dist);

app_sf_freq = fft(approx_sf);
app_tf_freq = fft(approx_tf);
exact_freq = fft(exact_sol);

app_sf_ps = log10(abs(fftshift(app_sf_freq.*conj(app_sf_freq)))/KT);
app_tf_ps = log10(abs(fftshift(app_tf_freq.*conj(app_sf_freq)))/KT);
exact_ps = log10(abs(fftshift(exact_freq.*conj(app_sf_freq)))/KT);
Kvals = -K+1:K;

%%%%%%%%%%%%%%%%%%%%
%plot_options = {'fontsize',16,'Interpreter','Latex'};
figure(1)
plot(Xvals,approx_sf,'k--',Xvals,approx_tf,'k-',Xvals,exact_sol,'b','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$x$','Interpreter','LaTeX','FontSize',30)
legend({['$\eta_{',num2str(Nplates/2),'p}(x,t_{f})$'],['$\eta_{',num2str(Nplates),'p}(x,t_{f})$'],'$\eta(x,t_{f})$'},'Interpreter','LaTeX')

figure(2)
plot(Kvals,app_sf_ps,'k--',Kvals,app_tf_ps,'k-',Kvals,exact_ps,'b','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$k$','Interpreter','LaTeX','FontSize',30)
legend({['$\eta_{',num2str(Nplates/2),'p}(k,t_{f})$'],['$\eta_{',num2str(Nplates),'p}(k,t_{f})$'],'$\eta(k,t_{f})$'},'Interpreter','LaTeX')

figure(3)
plot(eraxis,sf_dist,'k--',eraxis,tf_dist,'k-','LineWidth',2)
h = set(gca,'FontSize',18);
set(h,'Interpreter','LaTeX')
xlabel('$\log_{10} \left|E_{a}\right|$','Interpreter','LaTeX','FontSize',30)
legend({['$\eta_{',num2str(Nplates/2),'p}(x,t_{f})$'],['$\eta_{',num2str(Nplates),'p}(x,t_{f})$']},'Interpreter','LaTeX')

figure(4)
plot(tvals,msqerror_sfloat,'k--',tvals,msqerror_tfloat,'k-','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$T_{s}$','Interpreter','LaTeX','FontSize',30)
ylabel('$\left|\left|\eta_{tr}(\cdot,t) - \bar{\eta}_{a}(\cdot,t)\right|\right|_{2}/\left|\left|\eta_{tr}(\cdot,t)\right|\right|_{2}$','Interpreter','LaTeX','FontSize',30)
legend({['$\eta_{',num2str(Nplates/2),'p}(x,t)$'],['$\eta_{',num2str(Nplates),'p}(x,t)$']},'Interpreter','LaTeX')
