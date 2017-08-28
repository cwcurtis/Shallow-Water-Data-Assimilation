function data_plotter(K,Llx,tf,dt,dts,Nens,sig,Nplates)

KT = 2*K;

Xfloats = linspace(-Llx,Llx,Nplates/2)';
[fin_dat_sfloat,tvals,msqerror_sfloat] = kalman_filter_experimenter(K,Llx,tf,dt,dts,Nens,sig,Xfloats);

Xfloats = linspace(-Llx,Llx,Nplates)';
[fin_dat_tfloat,tvals,msqerror_tfloat] = kalman_filter_experimenter(K,Llx,tf,dt,dts,Nens,sig,Xfloats);

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

figure(1)
plot(Xvals,approx_sf,'k--',Xvals,approx_tf,'k-',Xvals,exact_sol,'b','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$x$','Interpreter','LaTeX','FontSize',30)
legend({'$\eta_{4p}(x,t_{f})$','$\eta_{8p}(x,t_{f})$','$\eta(x,t_{f})$'},'Interpreter','LaTeX')

figure(2)
plot(Xvals,sf_error,'k--',Xvals,tf_error,'k-','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$x$','Interpreter','LaTeX','FontSize',30)
ylabel('$\left|\eta(x,t_{f})-\eta_{a}(x,t_{f})\right|$','Interpreter','LaTeX','FontSize',30)
legend({'$\eta_{4p}(x,t_{f})$','$\eta_{8p}(x,t_{f})$'},'Interpreter','LaTeX')

figure(3)
plot(eraxis,sf_dist,'k--',eraxis,tf_dist,'k-','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$\log_{10} \left|E_{a}\right|$','Interpreter','LaTeX','FontSize',30)
legend({'$\eta_{4p}(x,t_{f})$','$\eta_{8p}(x,t_{f})$'},'Interpreter','LaTeX')

figure(4)
plot(tvals,msqerror_sfloat,'k--',tvals,msqerror_tfloat,'k-','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$T_{s}$','Interpreter','LaTeX','FontSize',30)
ylabel('$\left|\left|\eta_{tr}(\cdot,t) - \bar{\eta}_{a}(\cdot,t)\right|\right|_{2}$','Interpreter','LaTeX','FontSize',30)
legend({'$\eta_{4p}(x,t)$','$\eta_{8p}(x,t)$'},'Interpreter','LaTeX')
