function data_plotter_temporal(K,Llx,tf,dt,dts,Nens,sig,Nplates)

dx = Llx/K;
Xfloats = linspace(-Llx,Llx-dx,Nplates)';

[smth_dat,path_dat,surf_dat,tvals,msqerror_sfloat] = kalman_filter_temporal(K,Llx,tf,dt,dts,Nens,sig,Xfloats);

figure(1)
plot(tvals,smth_dat(1,:),'k--',tvals,path_dat(1,:),'b--',tvals,surf_dat(1,:),'k-','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$t$','Interpreter','LaTeX','FontSize',30)
legend({'$\eta_{s}(x_{f},t)$','$\eta_{d}(x_{f},t)$','$\eta(x_{f},t)$'},'Interpreter','LaTeX')

%figure(2)
%plot(eraxis,sf_dist,'k--',eraxis,tf_dist,'k-','LineWidth',2)
%h = set(gca,'FontSize',30);
%set(h,'Interpreter','LaTeX')
%xlabel('$\log_{10} \left|E_{a}\right|$','Interpreter','LaTeX','FontSize',30)
%legend({'$\eta_{4p}(x,t_{f})$','$\eta_{8p}(x,t_{f})$'},'Interpreter','LaTeX')

figure(3)
plot(tvals,msqerror_sfloat,'k--','LineWidth',2)
h = set(gca,'FontSize',30);
set(h,'Interpreter','LaTeX')
xlabel('$T_{s}$','Interpreter','LaTeX','FontSize',30)
ylabel('$\left|\left|\eta_{tr}(\cdot,t) - \bar{\eta}_{a}(\cdot,t)\right|\right|_{2}$','Interpreter','LaTeX','FontSize',30)
legend({'$\eta_{4p}(x,t)$','$\eta_{8p}(x,t)$'},'Interpreter','LaTeX')
