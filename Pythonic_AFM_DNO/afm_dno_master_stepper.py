import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import afm_dno_solver_suit as afm_dno
import afm_path_plotter_suit as afm_path
import time

K = 128
Llx = 10
tf = 1.
dt = 1e-2
nsteps = int(np.floor(tf/dt))
KT = 2*K
dx = 2.*Llx/KT
X = np.arange(-Llx, Llx, dx)

ep = .1
mu = np.sqrt(ep)
Mval = 14

usolrk4 = pyfftw.empty_aligned(KT,dtype='float64')
usolimp = pyfftw.empty_aligned(KT,dtype='float64')
path = np.zeros([2,nsteps],dtype='float64')
xint = np.zeros(2,dtype='float64')
xint[0] = 0.
xint[1] = ep*(np.cos(np.pi * xint[0] / Llx))

t0 = time.time()
#usolrk4[:] = afm_dno.afm_dno_solver_rk4(K, ep, mu, Llx, tf, Mval, dt)
#usolimp[:] = afm_dno.afm_dno_solver_imex(K, ep, mu, Llx, tf, Mval, dt)
path[:,:] = afm_path.kalman_filter(K, ep, mu, Llx, Mval, xint, dt, tf)
t1 = time.time()

print t1-t0

#plt.plot(X,ep*usolrk4,color='k',ls="-")
plt.plot(path[0,:],path[1,:],color='b',ls="--")
#plt.xlim(-.00005,.0)
#plt.ylim(.08,.12)
plt.show()