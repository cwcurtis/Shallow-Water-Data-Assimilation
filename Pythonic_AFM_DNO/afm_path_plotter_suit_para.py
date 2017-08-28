import pyfftw 
import numpy as np
from numpy.matlib import repmat
from afm_dno_solver_suit import nonlinearity

###############################################################################
###############################################################################

def phi_eval(params, xvec, eta, q, Kmesh, mDk, tnh, fft):
    
    K,ep,mu,Mval = params[0],params[1],params[2],params[4]
        
    KT = 2*K
    xval = xvec[0].real
    zval = xvec[1].real        
        
    phis = np.zeros([KT,Mval+1],dtype='complex128')
        
    etap = pyfftw.empty_aligned(KT,dtype='float64')
    etapow = pyfftw.empty_aligned(KT,dtype='float64')
    Dkpow = pyfftw.empty_aligned(KT,dtype='float64')
    cvec = pyfftw.empty_aligned(KT,dtype='float64')
    tnvec = pyfftw.empty_aligned(KT,dtype='float64')
    ovec = pyfftw.empty_aligned(KT,dtype='float64')
    evec = pyfftw.empty_aligned(KT,dtype='complex128')
    phif = pyfftw.empty_aligned(KT,dtype='complex128')    
    
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

def solver_rk4(params, L1, Kmesh, mDk, tnh, 
                       eLdt, eLdt2, uint, Ndat, fft):
    
    K,ep,mu,Mval,dt = params[0],params[1],params[2],params[4],params[5]
        
    KT = 2 * K
    # Find the wave numbers to implement the 2/3 de-aliasing throughout
    Kc = int(np.floor(2. * K / 3.))
    Kuc = KT - Kc + 1
    Kc = Kc + 1
    
    #Xmesh = np.linspace(-Llx,Llx,KT,endpoint=False)
    eta_in = pyfftw.empty_aligned(KT,dtype='float64')
    q_in = pyfftw.empty_aligned(KT,dtype='float64')
    
    eta_out = np.zeros(KT,dtype='float64')
    q_out = np.zeros(KT,dtype='float64')

    etan = pyfftw.empty_aligned(KT,dtype='complex128')
    qn = pyfftw.empty_aligned(KT,dtype='complex128')
    G0 = pyfftw.empty_aligned(KT,dtype='float64')
    etah2 = pyfftw.empty_aligned(KT,dtype='complex128')
    qh2 = pyfftw.empty_aligned(KT,dtype='complex128')
    ufreq = pyfftw.empty_aligned(2*KT,dtype='complex128')
   
    k1 = pyfftw.empty_aligned(2*KT,dtype='complex128')
    k2 = pyfftw.empty_aligned(2*KT,dtype='complex128')
    k3 = pyfftw.empty_aligned(2*KT,dtype='complex128')
    k4 = pyfftw.empty_aligned(2*KT,dtype='complex128')
    svec = pyfftw.empty_aligned(2*KT,dtype='complex128')
    
    k1t = np.zeros(2*Ndat,dtype='float64')
    k2t = np.zeros(2*Ndat,dtype='float64')
    k3t = np.zeros(2*Ndat,dtype='float64')
    k4t = np.zeros(2*Ndat,dtype='float64')
    xvi = np.zeros(2*Ndat,dtype='float64')
    xfv = np.zeros(2*Ndat,dtype='float64')
    
    ###########################################################################
    eta_in[:] = uint[:KT]    
    q_in[:] = uint[KT:2*KT]
    
    etan[:] = fft.fft(eta_in)
    qn[:] = fft.fft(q_in)
    xvi[:] = uint[2*KT:]
    
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
    
    eta_out[:] = fft.ifft(etan).real
    q_out[:] = fft.ifft(qn).real
    
    return np.concatenate((np.concatenate((eta_out,q_out),axis=0),xfv),axis=0)

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

    xfluc = np.zeros([2*(KT+Ndat),Nens],dtype='float64')
    Pmat = np.zeros([2*(KT+Ndat),2*Ndat],dtype='float64')
    Pdd = np.zeros([2*Ndat,2*Ndat],dtype="float64")
    xmean = np.zeros(2*(KT+Ndat),dtype='float64')
    mrhs = np.zeros([2*Ndat,Nens],dtype='float64')
                         
    Idd = np.identity(2*Ndat,dtype='float64')
    
    xmean = np.sum(xf,axis=1)/Nens
    xfluc[:,:] = xf - repmat(xmean,Nens,1).T
    
    Pmat[:,:] = np.matmul(xfluc,xfluc[2*KT:,:].T)/(Nens - 1.)
    
    Pdd[:,:] = Pmat[2*KT:,:].real
    mrhs[:,:] = np.linalg.solve(Pdd + (sig**2.)*Idd, dmat-xf[2*KT:,:]).real
    
    return xf + np.matmul(Pmat,mrhs) 
    
###############################################################################
###############################################################################

def data_stream_maker(params,un,L1,Kmesh,mDk,tnh,eLdt,eLdt2,tf,nindt,
                      nmax,path_dat,surf_dat,Ndat,fft):

    K = params[0]
    
    KT = 2*K
    nsamp = int(np.round(nmax/nindt))

    unloc = np.zeros(2*(KT+Ndat),dtype='float64')
    unloc[:] = un    
    
    path_dat = np.zeros([2*Ndat,nsamp],dtype='float64')
    path_dat[:,0] = unloc[2*KT:]
    dcnt = 1
    
    for jj in xrange(1,nmax):
        unloc[:] = solver_rk4(params, L1, Kmesh, mDk, tnh, 
                                     eLdt, eLdt2, unloc, Ndat, fft)
                       
        if jj%nindt == 0:
            path_dat[:,dcnt] = unloc[2*KT:]
            surf_dat[:,dcnt] = unloc[:KT]                        
            dcnt += 1