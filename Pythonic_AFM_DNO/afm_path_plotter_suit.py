import numpy as np
import my_fft

from pyfftw import empty_aligned
from numpy.matlib import repmat

from afm_dno_solver_suit import nonlinearity
from scipy.linalg import expm

#import matplotlib.pyplot as plt

###############################################################################
###############################################################################

def phi_eval(params, xvec, eta, q, Kmesh, mDk, tnh, fft):
    
    K,ep,mu,Mval = params[0],params[1],params[2],params[4]
        
    KT = 2*K
    xval = xvec[0].real
    zval = xvec[1].real        
        
    phis = np.zeros([KT,Mval+1],dtype='complex128')
        
    etap = empty_aligned(KT,dtype='float64')
    etapow = empty_aligned(KT,dtype='float64')
    Dkpow = empty_aligned(KT,dtype='float64')
    cvec = empty_aligned(KT,dtype='float64')
    tnvec = empty_aligned(KT,dtype='float64')
    ovec = empty_aligned(KT,dtype='float64')
    evec = empty_aligned(KT,dtype='complex128')
    phif = empty_aligned(KT,dtype='complex128')    
    
    evec[:] = np.exp(1j*Kmesh*xval)
    ovec[:] = np.ones(KT,dtype='float64')
    tnvec[:] = np.tanh(mDk*zval)
    cvec[:] = np.cosh(mDk*zval)    
    etap[:] = fft.ifft(eta).real
   
    phis[:,0] = q
    phif[:] = q    
    eppow = ep
    
    for ll in xrange(1, Mval+1):
        etapow[:] = ovec
        Dkpow[:] = ovec
    
        for jj in xrange(1,ll + 1):
            etapow *= mu*etap/jj
            Dkpow *= Kmesh

            if jj%2 == 0:
                phis[:,ll] += etapow*fft.ifft(Dkpow*phis[:,ll-jj]).real                                
            else:
                phis[:,ll] += etapow*fft.ifft(tnh*Dkpow*phis[:,ll-jj]).real                                                
            
        phis[:,ll] = -fft.fft(phis[:,ll])            
        phif[:] += eppow*phis[:,ll]    
        eppow *= ep        
    
    phif *= -cvec #Note, the "-" is to correct for pyfftw conventions.

    return ep/KT*np.array([np.sum(1j*Kmesh*phif*(ovec+tnvec*tnh)*evec).real, 
                           np.sum(mDk/mu*phif*(tnvec/mu + tnh/mu)*evec).real])                     
                               
    
###############################################################################
###############################################################################

def afm_dno_solver_rk4(params, L1, Kmesh, mDk, tnh, 
                       eLdt, eLdt2, uint, xvi, Ndat, fft):
    
    K,ep,mu,Mval,dt = params[0],params[1],params[2],params[4],params[5]
        
    KT = 2 * K
    # Find the wave numbers to implement the 2/3 de-aliasing throughout
    Kc = int(np.floor(2. * K / 3.))
    Kuc = KT - Kc + 1
    Kc = Kc + 1
    
    #Xmesh = np.linspace(-Llx,Llx,KT,endpoint=False)
    
    etan = empty_aligned(KT,dtype='complex128')
    qn = empty_aligned(KT,dtype='complex128')
    etap = empty_aligned(KT,dtype='float64')
    qp = empty_aligned(KT,dtype='float64')
    G0 = empty_aligned(KT,dtype='float64')
    etah2 = empty_aligned(KT,dtype='complex128')
    qh2 = empty_aligned(KT,dtype='complex128')
    ufreq = empty_aligned(2*KT,dtype='complex128')
    
    k1 = empty_aligned(2*KT,dtype='complex128')
    k2 = empty_aligned(2*KT,dtype='complex128')
    k3 = empty_aligned(2*KT,dtype='complex128')
    k4 = empty_aligned(2*KT,dtype='complex128')
    svec = empty_aligned(2*KT,dtype='complex128')
    
    k1t = np.zeros(2*Ndat,dtype='float64')
    k2t = np.zeros(2*Ndat,dtype='float64')
    k3t = np.zeros(2*Ndat,dtype='float64')
    k4t = np.zeros(2*Ndat,dtype='float64')
    xfv = np.zeros(2*Ndat,dtype='float64')
    
    ###########################################################################
    etan[:] = fft.fft(uint[:KT])
    qn[:] = fft.fft(uint[KT:])
    
    etan[Kc-1:Kuc] = 0.
    qn[Kc-1:Kuc] = 0.
    ufreq[:] = np.concatenate((etan,qn))
    
    G0[:] = fft.ifft(L1*qn).real

    k1[:] = dt*nonlinearity(K,etan,qn,G0,ep,mu,Kmesh,mDk,tnh,Mval,fft)

    for jj in xrange(Ndat):
        k1t[2*jj:2*jj+2] = dt*phi_eval(params,xvi[2*jj:2*jj+2],etan,qn,Kmesh,
                                       mDk,tnh,fft)
                                       
    ###########################################################################
    svec[:] = np.matmul(eLdt2, (ufreq + .5*k1))

    etah2[:] = svec[:KT]
    qh2[:] = svec[KT:]
    etah2[Kc-1:Kuc] = 0.
    qh2[Kc-1:Kuc] = 0.

    G0[:] = fft.ifft(L1*qh2).real

    k2[:] = dt*nonlinearity(K,etah2,qh2,G0,ep,mu,Kmesh,mDk,tnh,Mval,fft)

    for jj in xrange(Ndat):
        k2t[2*jj:2*jj+2] = dt*phi_eval(params,
                                       xvi[2*jj:2*jj+2]+k1t[2*jj:2*jj+2]/2.,
                                       etah2,qh2,Kmesh,mDk,tnh,fft)
                                       
    ###########################################################################
    svec[:] = np.matmul(eLdt2, ufreq) + .5*k2

    etah2[:] = svec[:KT]
    qh2[:] = svec[KT:]
    etah2[Kc-1:Kuc] = 0.
    qh2[Kc-1:Kuc] = 0.

    G0[:] = fft.ifft(L1*qh2).real

    k3[:] = dt*nonlinearity(K,etah2,qh2,G0,ep,mu,Kmesh,mDk,tnh,Mval,fft)

    for jj in xrange(Ndat):
        k3t[2*jj:2*jj+2] = dt*phi_eval(params,
                                       xvi[2*jj:2*jj+2]+k2t[2*jj:2*jj+2]/2.,
                                       etah2,qh2,Kmesh,mDk,tnh,fft)
                                       
    ###########################################################################
    svec[:] = np.matmul(eLdt, ufreq) + np.matmul(eLdt2, k3)

    etah2[:] = svec[:KT]
    qh2[:] = svec[KT:]
    etah2[Kc-1:Kuc] = 0.
    qh2[Kc-1:Kuc] = 0.

    G0[:] = fft.ifft(L1*qh2).real

    k4[:] = dt*nonlinearity(K,etah2,qh2,G0,ep,mu,Kmesh,mDk,tnh,Mval,fft)

    for jj in xrange(Ndat):
        k4t[2*jj:2*jj+2] = dt*phi_eval(params,
                                       xvi[2*jj:2*jj+2]+k3t[2*jj:2*jj+2],
                                       etah2,qh2,Kmesh,mDk,tnh,fft)
                                       
    ###########################################################################
    ufreq[:] = (np.matmul(eLdt,(ufreq+k1/6.))
             + np.matmul(eLdt2,(k2+k3)/3.)+k4/6.)

    xfv[:] = (xvi+(k1t+2.*(k2t+k3t)+k4t)/6.).real

    etan[:] = ufreq[:KT]
    qn[:] = ufreq[KT:]

    etan[Kc-1:Kuc] = 0.
    qn[Kc-1:Kuc] = 0.
    etap[:] = fft.ifft(etan).real
    qp[:] = fft.ifft(qn).real
    
    return np.concatenate((np.concatenate((etap,qp),axis=0),xfv),axis=0)

###############################################################################
###############################################################################

def data_build(Ndat,Nens,pathd,sig):
    
    dmat = np.zeros([2*Ndat,Nens],dtype='float64')
    for jj in xrange(Nens):
        dmat[:,jj] = pathd + sig*np.random.standard_normal(2*Ndat)
    return dmat
    
###############################################################################
###############################################################################

def analysis_step(K,Nens,Ndat,xf,dmat,sig):    
    
    # xf - matrix representing ensemble of forecasts.  
    # Nens - number of members in the ensemble
    # Ndat - number of data values sampled 
    # dmat - data matrix at time t_k
    # sig - standard deviation of Gaussian distribution for errors

    KT = 2*K

    xfluc = np.concatenate(
        (repmat(empty_aligned(2*KT,dtype='float64'),Nens,1).T,
         np.zeros([2*Ndat,Nens],dtype='float64')))    
    
    Pmat = np.concatenate((np.zeros([2*KT,2*Ndat],dtype="float64"),
                           np.zeros([2*Ndat,2*Ndat],dtype="float64"))) 
    
    Pdd = np.zeros([2*Ndat,2*Ndat],dtype="float64")
    
    xmean = np.concatenate((empty_aligned(2*KT,dtype='float64'),
                            np.zeros(2*Ndat,dtype='float64')))
    
    Idd = np.identity(2*Ndat,dtype='float64')
    mrhs = np.zeros([2*Ndat,Nens],dtype='float64')
    
    xmean = np.sum(xf,axis=1)/Nens
    xfluc[:,:] = xf - repmat(xmean,Nens,1).T
    
    Pmat[:,:] = np.matmul(xfluc,xfluc[2*KT:,:].T)/(Nens - 1.)
    
    Pdd[:,:] = Pmat[2*KT:,:].real
    mrhs[:,:] = np.linalg.solve(Pdd + (sig**2.)*Idd, dmat-xf[2*KT:,:]).real
    
    return xf + np.matmul(Pmat,mrhs) 
    
###############################################################################
###############################################################################

def data_stream_maker(params,xint,un,L1,Kmesh,mDk,tnh,eLdt,eLdt2,tf,nindt,
                      nmax,path_dat,surf_dat,Ndat,fft):

    K = params[0]
    
    KT = 2*K
    nsamp = int(np.round(nmax/nindt))

    svec = np.concatenate((empty_aligned(2*KT, dtype='float64'),
                           np.zeros(2*Ndat,dtype='float64')))
    
    unloc = empty_aligned(2*KT,dtype='float64')
 
    path = xint    
    path_dat = np.zeros([2*Ndat,nsamp],dtype='float64')
    path_dat[:,0] = xint
    dcnt = 1
    unloc[:] = un
    
    for jj in xrange(1,nmax):
        svec[:] = afm_dno_solver_rk4(params, L1, Kmesh, mDk, tnh, 
                                     eLdt, eLdt2, unloc, path, Ndat, fft)
                       
        unloc[:] = svec[:2*KT]
        path = svec[2*KT:]
        
        if jj%nindt == 0:
            path_dat[:,dcnt] = path
            surf_dat[:,dcnt] = unloc[:KT]                        
            dcnt += 1
    
###############################################################################
###############################################################################

def kalman_filter(params,xint,tf):

    K,mu,Llx = params[0],params[2],params[3]
    dt,dts,Nens,sig = params[5],params[6],params[7],params[8]
        
    KT = 2*K
    # Find the wave numbers to implement the 2/3 de-aliasing throughout

    nindt = int(np.round(dts/dt))
    nmax = int(np.round(tf/dt))
    nsamp = int(np.round(nmax/nindt))
    Ndat = xint.size/2
        
    fft = my_fft.my_fft(KT)
    
    Xmesh = empty_aligned(KT,dtype='float64')
    eta0 = empty_aligned(KT, dtype='float64')
    q0 = empty_aligned(KT, dtype='float64')
    Kmesh = empty_aligned(KT, dtype='float64')
    tnh = empty_aligned(KT, dtype='float64')
    mDk = empty_aligned(KT, dtype='float64')
    L1 = empty_aligned(KT, dtype='float64')
    un = empty_aligned(2*KT, dtype='float64')

    xapprox = empty_aligned(KT, dtype='float64')
    xtrue = empty_aligned(KT, dtype='float64')

    x0 = np.concatenate((empty_aligned(2*KT,dtype='float64'),
                         np.zeros(2*Ndat,dtype='float64')))

    eLdt2 = np.zeros([2*KT,2*KT],dtype='float64')
    eLdt = np.zeros([2*KT,2*KT],dtype='float64')
    eLdt2s = np.zeros([2*KT,2*KT],dtype='float64')
    eLdts = np.zeros([2*KT,2*KT],dtype='float64')
        
    dmat = np.zeros([2*Ndat,Nens],dtype='float64')
    
    # Build forcast and analysis matrices.
    xf = np.concatenate(
        (repmat(empty_aligned(2*KT,dtype='float64'),Nens,1).T,
         np.zeros([2*Ndat,Nens],dtype='float64')))    
    
    xa = np.concatenate(
        (repmat(empty_aligned(2*KT,dtype='float64'),Nens,1).T,
         np.zeros([2*Ndat,Nens],dtype='float64')))    

    # Build time invariant vectors and matrices associated with model 
    # computations.  
    Xmesh[:] = np.linspace(-Llx, Llx, KT, endpoint=False)

    Kmesh[:] = np.pi/Llx*(
        np.concatenate((np.arange(0,K+1),np.arange(-K+1,0)),0))
    L1[:] = Kmesh*np.tanh(mu*Kmesh)/mu
    mDk[:] = mu*Kmesh
    tnh[:] = np.tanh(mDk)

    eta0[:] = np.cos(np.pi*Xmesh/Llx)
    q0[:] = np.sin(np.pi*Xmesh/Llx)
    un[:] = np.concatenate((eta0,q0))

    Zs = np.zeros([KT, KT], dtype='float64')
    Is = np.identity(KT, dtype='float64')
    Lop = np.concatenate((np.concatenate((Zs, np.diag(L1)), 1),
                          np.concatenate((-Is, Zs), 1)),0)
    
    eLdt2 = expm(dt*Lop/2.)
    eLdt = np.matmul(eLdt2,eLdt2)
    
    eLdt2s = expm(dts*Lop/2.)
    eLdts = np.matmul(eLdt2s,eLdt2s)
        
    # Build initial ensemble of forecasts
    x0[:] = np.concatenate((un,xint))        
    for ll in xrange(Nens):
        xf[:,ll] = x0 + sig*np.random.standard_normal(2*(KT+Ndat))
    
    # Storage for "true" path and surface.
    path_dat = np.zeros([2*Ndat,nsamp],dtype='float64')
    surf_dat = repmat(empty_aligned(KT,dtype='float64'),nsamp,1).T
    
    # Build "true" data paths
    data_stream_maker(params,xint,un,L1,Kmesh,mDk,tnh,eLdt,eLdt2,
                      tf,nindt,nmax,path_dat,surf_dat,Ndat,fft)
   
    # Run the filter forward in time.                        
    for jj in xrange(nsamp):
        dmat[:,:] = data_build(Ndat,Nens,path_dat[:,jj],sig)
        xa[:,:] = analysis_step(K,Nens,Ndat,xf,dmat,sig)   
        for kk in xrange(Nens):            
            xf[:,kk] = afm_dno_solver_rk4(params,L1,Kmesh,mDk,tnh,eLdts,eLdt2s,
                                          xa[:2*KT,kk],xa[2*KT:,kk],Ndat,fft)
                                     
    # Average to get final approximation.                                       
    xapprox[:] = np.sum(xf[:KT,:],axis=1)/Nens
    xtrue[:] = surf_dat[:,nsamp-1]          
    return np.concatenate((xapprox,xtrue))