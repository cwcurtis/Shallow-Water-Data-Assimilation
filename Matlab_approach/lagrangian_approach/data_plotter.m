function data_plotter(K,Llx,tf,dt,dts,Nens,sig)

KT = 2*K;

Xfloats = [0];
fin_dat_sfloat = kalman_filter_experimenter(K,Llx,tf,dt,dts,Nens,sig,Xfloats);

Xfloats = linspace(-Llx,Llx,24)';
fin_dat_tfloat = kalman_filter_experimenter(K,Llx,tf,dt,dts,Nens,sig,Xfloats);

approx_sf = fin_dat_sfloat(1:KT);
exact_sol = fin_dat_sfloat(KT+1:2*KT);
approx_tf = fin_dat_tfloat(1:KT);

dx = Llx/K;
Xvals = (-Llx:dx:Llx-dx)';

figure(1)
plot(Xvals,approx_sf,'k-',Xvals,approx_tf,'k--',Xvals,exact_sol,'b','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$x$','Interpreter','LaTeX','FontSize',30)
legend({'$\eta_{st}(x,t_{f})$','$\eta_{mt}(x,t_{f})$','$\eta(x,t_{f})$'},'Interpreter','LaTeX')

figure(2)
plot(Xvals,log10(abs(approx_sf-exact_sol)),'k-',Xvals,log10(abs(approx_tf-exact_sol)),'k--','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$x$','Interpreter','LaTeX','FontSize',30)
ylabel('$\left|\eta(x,t_{f})-\eta_{a}(x,t_{f})\right|$','Interpreter','LaTeX','FontSize',30)
