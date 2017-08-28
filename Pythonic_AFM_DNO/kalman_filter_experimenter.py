import matplotlib.pyplot as plt
import numpy as np
import my_fft
import afm_path_plotter_suit_para as afm
import multiprocessing as mp

from scipy.linalg import expm
from pyfftw import empty_aligned

if __name__=='__main__':

    K = 64
    KT = 2*K

    Llx = 10
    tf = 2e-2
    dt = 1e-2
    nsteps = int(np.floor(tf/dt))

    dts = 1e-2
    Nens = 8
    sig = 1.

    Nproc = 8
    pool = mp.Pool(processes=Nproc)
    
    Xmesh = np.linspace(-Llx,Llx,KT,endpoint=False)

    ep = .1
    mu = np.sqrt(ep)
    Mval = 14

    params = [K,ep,mu,Llx,Mval,dt,dts,Nens,sig]

    Xfloats = np.array([-.1,.1])
    xint = np.zeros(2*Xfloats.size,dtype='float64')

    xint[0:2*Xfloats.size:2] = Xfloats
    xint[1:2*Xfloats.size:2] = ep*(np.cos(np.pi * Xfloats / Llx))
     
    nindt = int(np.round(dts/dt))
    nmax = int(np.round(tf/dt))
    nsamp = int(np.round(nmax/nindt))
    Ndat = xint.size/2
        
    fft = my_fft.my_fft(KT)
    
    Kmesh = empty_aligned(KT, dtype='float64')
    tnh = empty_aligned(KT, dtype='float64')
    mDk = empty_aligned(KT, dtype='float64')
    L1 = empty_aligned(KT, dtype='float64')
    
    # Build forcast and analysis matrices.
    xf = np.zeros([2*(KT+Ndat),Nens],dtype='float64')        
    
    # Build time invariant vectors and matrices associated with model 
    # computations.  
    Xmesh = np.linspace(-Llx, Llx, KT, endpoint=False)

    Kmesh[:] = np.pi/Llx*(
        np.concatenate((np.arange(0,K+1),np.arange(-K+1,0)),0))
    L1[:] = Kmesh*np.tanh(mu*Kmesh)/mu
    mDk[:] = mu*Kmesh
    tnh[:] = np.tanh(mDk)

    eta0 = np.cos(np.pi*Xmesh/Llx)
    q0 = np.sin(np.pi*Xmesh/Llx)
    un = np.concatenate((np.concatenate((eta0,q0)),xint))

    Zs = np.zeros([KT, KT], dtype='float64')
    Is = np.identity(KT, dtype='float64')
    Lop = np.concatenate((np.concatenate((Zs, np.diag(L1)), 1),
                          np.concatenate((-Is, Zs), 1)),0)
    
    eLdt2 = expm(dt*Lop/2.)
    eLdt = np.matmul(eLdt2,eLdt2)
    
    eLdt2s = expm(dts*Lop/2.)
    eLdts = np.matmul(eLdt2s,eLdt2s)
        
    # Build initial ensemble of forecasts
    for ll in xrange(Nens):
        xf[:,ll] = un + sig*np.random.standard_normal(2*(KT+Ndat))
    
    # Storage for "true" path and surface.
    path_dat = np.zeros([2*Ndat,nsamp],dtype='float64')
    surf_dat = np.zeros([KT,nsamp],dtype='float64')
    
    # Build "true" data paths
    afm.data_stream_maker(params,un,L1,Kmesh,mDk,tnh,eLdt,eLdt2,
                      tf,nindt,nmax,path_dat,surf_dat,Ndat,fft)
    
    # Run the filter forward in time.                        
    for jj in xrange(nsamp):
        dmat = afm.data_build(Ndat,Nens,path_dat[:,jj],sig)
        xa = afm.analysis_step(K,Nens,Ndat,xf,dmat,sig)   
    
        # Set up parallel processing 
        results = [pool.apply_async(afm.solver_rk4,
                                    args=(params,L1,Kmesh,mDk,tnh,eLdts,eLdt2s,
                                          col,Ndat,fft)) for col in xa.T]
        
        #results = [p.get() for p in results]        
        
        #print results         
        #lcnt = 0                                  
        #for p in results:
        #    xf[:,lcnt] = p.get()
        #    print "We get here?"            
        #    lcnt += 1

               
    # Average to get final approximation.                                       
    xapprox = np.sum(xf[:KT,:],axis=1)/Nens

    plt.plot(Xmesh,xapprox,color='k',ls="-")
    plt.plot(Xmesh,surf_dat[:,nsamp-1],color='b',ls="--")
    plt.show()