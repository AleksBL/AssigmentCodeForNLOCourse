import numpy as np
from scipy.optimize import fsolve, root, newton, minimize
import matplotlib.pyplot as plt
from scipy.special import jv, kv
from scipy.constants import mu_0,epsilon_0,c
import scipy
from scipy import linalg, matrix
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, interp1d
from numba import njit
import scipy.fftpack as sft




def null(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)
def null2(a):
    b = a[:, 0].copy()
    x = np.linalg.lstsq(a[:, 1:], -b)[0]
    x = np.r_[1, x]
    x /= np.linalg.norm(x)
    return x

def null3(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()

def constraint(x):
    return np.linalg.norm(x)-1
cons = {'type':'eq', 'fun': constraint}

def null4(a):
    def f(x):
        xr = x[0:4]
        xi = x[4: ]
        X = xr + 1j * xi
        return np.linalg.norm(a.dot(X))
    
    x0 = null2(a)
    R, I = x0.real, x0.imag
    V = np.hstack((R,I))
    sol = minimize(f,V, constraints= (cons), method = 'SLSQP', options = {'ftol':1e-10, 'disp':True, 
                                                                          'maxiter':10**3})
    out = sol.x[0:4]+1j * sol.x[4:]
    
    return out


ep0 = epsilon_0
mu0 = mu_0
pi = np.pi



def KP(k0, n, beta):
    out = np.sqrt(k0**2 * n**2  - beta**2 + 0j)
    return out

def GP(k0,n,beta):
    return -1j * KP(k0, n, beta)

def diff_jv(m,x):
    return  0.5*(jv(m-1,x)  - jv(m+1, x))

def diff_kv(m,x):
    return -0.5*(kv(m-1,x)  + kv(m+1, x))

def eq_system(beta, a, n1, n2, k0, m):
    nb  = len(beta)
    M   = np.zeros((nb, 4,4), dtype = np.complex128)
    
    k = np.sqrt(n1**2 * k0**2 - (beta**2) + 0j)#KP(k0, n1, beta)
    g = np.sqrt(beta**2 - (n2**2 * k0**2) + 0j)#GP(k0, n2, beta)
    
    
    
    kp1 =       k      
    kp2 =  1j * g
    gp2 =       g
    #print(gp2, kp1)
    
    #print(kp1,kp2)
    
    omega = c*k0
    
    M[:,0,0] =    jv(m,  kp1 * a)
    M[:,0,1] =  - kv(m,  gp2 * a)
    
    M[:,1,2] =    jv(m,  kp1 * a)
    M[:,1,3] =  - kv(m,  gp2 * a)
    
    M[:,2,0] =   (1j / kp1**2)*(beta/a) * 1j * m     * jv(m,kp1*a)
    M[:,2,1] =  -(1j / kp2**2)*(beta/a) * 1j * m     * kv(m,gp2*a)
    M[:,2,2] =  -(1j / kp1**2)*  mu0         * omega * diff_jv(m,kp1 * a) * kp1
    M[:,2,3] =  +(1j / kp2**2)*  mu0         * omega * diff_kv(m,gp2 * a) * gp2
    
    M[:,3,0] =   (1j / kp1**2)*  ep0         * omega * n1**2 * diff_jv(m,kp1 * a) * kp1
    M[:,3,1] =  -(1j / kp2**2)*  ep0         * omega * n2**2 * diff_kv(m,gp2 * a) * gp2
    M[:,3,2] =   (1j / kp1**2)*(beta/a) * 1j * m     * jv(m,kp1*a)
    M[:,3,3] =  -(1j / kp2**2)*(beta/a) * 1j * m     * kv(m,gp2*a)
    return M

def eq_det(beta, a,n1,n2,k0,m):
    return np.linalg.det(eq_system(beta, a, n1, n2, k0, m))

def eq_det_scalar(beta,a,n1,n2,k0,m):
    BETA = np.array([beta])
    return eq_det(BETA, a,n1,n2,k0,m)

def Søn_f(beta, a, n1, n2, k0, m):
    k = np.sqrt(n1**2 * k0**2 - (beta**2))#KP(k0, n1, beta)
    g = np.sqrt(beta**2 - (n2**2 * k0**2))#GP(k0, n2, beta)
    f1 = diff_jv(m, k * a)  / (k *  jv(m,k * a))
    f2 = diff_kv(m, g * a)  / (g *  kv(m,g * a))
    
    t1 = f1 + f2 
    t2 = f1 + (n2/n1)**2 * f2
    
    t3 = (m*beta/(n1*k0*a))**2 * (1/(k**2) + 1/(g**2))**2
    return t1 * t2 - t3

def Field(rho, phi, A, B, C, D, beta, a, n1, n2, k0, m):
    F = np.zeros((len(rho), 6), dtype = np.complex128)
    idx_1  = np.where(rho< a)
    idx_2  = np.where(rho>=a)
    rho_1  = rho[idx_1]
    rho_2  = rho[idx_2]
    
    
    k = np.sqrt(n1**2 * k0**2 - (beta**2) + 0j)#KP(k0, n1, beta)
    g = np.sqrt(beta**2 - (n2**2 * k0**2)+ 0j)#GP(k0, n2, beta)
    k2 =  1j * g
    omega = c*k0
    
    
    Ez = np.zeros(rho.shape,dtype = np.complex128)
    Hz = np.zeros(rho.shape,dtype = np.complex128)
    dEz = np.zeros(rho.shape,dtype = np.complex128)
    dHz = np.zeros(rho.shape,dtype = np.complex128)
    
    
    E_phi = np.zeros(rho.shape,dtype = np.complex128)
    H_phi = np.zeros(rho.shape,dtype = np.complex128)
    
    E_rho = np.zeros(rho.shape,dtype = np.complex128)
    H_rho = np.zeros(rho.shape,dtype = np.complex128)
    
    Angular_part = np.exp(1j * m * phi)
    
    Ez[idx_1] = A * jv(m,k * rho_1)    #np.hstack((A * jv(m,k * rho_1) , B * kv(m,g * rho_2))) 
    Ez[idx_2] = B * kv(m,g * rho_2)
    
    
    Hz[idx_1] = C * jv(m, k * rho_1) #np.hstack((C * jv(m,k * rho_1) , D * kv(m,g * rho_2))) 
    Hz[idx_2] = D * kv(m, g * rho_2)
    
    dEz[idx_1] = A * diff_jv(m, k * rho_1) * k  # np.hstack((A * diff_jv(m, k * rho_1) * k, B * diff_kv(m, g * rho_2) * g)) * Angular_part
    dEz[idx_2] = B * diff_kv(m, g * rho_2) * g
    
    dHz[idx_1] = C * diff_jv(m, k * rho_1) * k  # np.hstack((C * diff_jv(m, k * rho_1) * k, D * diff_kv(m, g * rho_2) * g)) * Angular_part
    dHz[idx_2] = D * diff_kv(m, g * rho_2) * g
    
    Ez  = Ez * Angular_part
    Hz  = Hz * Angular_part
    dEz = dEz* Angular_part
    dHz = dHz* Angular_part
    
    E_phi_1 = (1j/k**2)*( (beta/rho_1) * 1j * m  * Ez[idx_1]   -  
                           mu0 * omega * dHz[idx_1]   )
    
    E_phi_2 = (1j/k2**2)*( (beta/rho_2) * 1j * m  * Ez[idx_2]   -  
                            mu0 * omega * dHz[idx_2]   )
    
    E_phi[idx_1] = E_phi_1
    E_phi[idx_2] = E_phi_2
    
    H_phi_1 = (1j/k**2) *( (beta/rho_1) * 1j * m  * Hz[idx_1]   +  
                            ep0 * n1**2 * omega * dEz[idx_1]     )
    
    H_phi_2 = (1j/k2**2)*( (beta/rho_2) * 1j * m  *  Hz[idx_2]   +  
                            ep0 * n2**2 * omega   * dEz[idx_2]     )
    
    H_phi[idx_1] = H_phi_1
    H_phi[idx_2] = H_phi_2
    
    E_rho_1 = (1j/k**2)  * ( beta * dEz[idx_1]  +
                             mu0  * omega/rho_1 * 1j * m * Hz[idx_1]  )
    
    E_rho_2 = (1j/k2**2) * ( beta * dEz[idx_2]  +
                             mu0  * omega/rho_2 * 1j * m * Hz[idx_2] )
    
    E_rho[idx_1] = E_rho_1
    E_rho[idx_2] = E_rho_2
    
    H_rho_1 = (1j/k**2) * ( beta / rho_1   * 1j * m *  Hz[idx_1] + 
                            ep0 * n1**2 *     omega * dEz[idx_1])
    H_rho_2 = (1j/k2**2) *( beta / rho_2   * 1j * m *  Hz[idx_2] + 
                            ep0 * n2**2 *     omega * dEz[idx_2])
    
    H_rho[idx_1] = H_rho_1
    H_rho[idx_2] = H_rho_2
    
    return Ez, Hz, E_phi, H_phi, E_rho, H_rho

class VarFiber:
    def __init__(self, n1,n2):
        self.n1 = n1
        self.n2 = n2
    
    def find_modes(self, Radii, Omegas, Modes, mode = '1'):
        nr = len(Radii)
        nf = len(Omegas)
        # A,B,C and D coefficients for the Electric and magnetic fields for the straight fibers
        Field_coeffs = np.full((nr, nf, len(Modes), 4),np.nan,dtype = np.complex128)
        Field_beta   = np.full((nr,nf, len(Modes)), np.nan, dtype = np.complex128)
        
        K0  = Omegas / c
        
        for im, M in enumerate(Modes):
            for ik, _k0 in enumerate(K0):
                
                b0 = np.array([K0[ik] *self.n2(Omegas[ik])])*1.000000001
                w = Omegas[ik]
                
                for ia, _a in enumerate(Radii):
                    if mode == '1':
                        def f(beta):
                            return eq_det(beta, _a,self.n1(w),self.n2(w),_k0,M)[0]
                    
                    elif mode == '2':
                        def f(beta):
                            return np.abs(Søn_f(beta, _a,self.n1(w),self.n2(w),_k0,M))
                    
                    x = fsolve(f, b0)
                    kmin, kmax = self.n2(w)*_k0 , self.n1(w) * _k0
                    
                    if kmin <= x < kmax:
                        Field_beta[ia, ik, im]      = x
                        Sys                         = eq_system(x, _a, self.n1(Omegas[ik]), 
                                                                self.n2(Omegas[ik]), 
                                                                _k0, 
                                                                M)[0]
                        
                        Field_coeffs[ia, ik, im, :] = null2(Sys)
                        b0 = x.copy()
        
        self.Field_coeffs = Field_coeffs.copy()
        self.Field_beta   = Field_beta.copy()
        self.Calced_Radii = Radii.copy()
        self.Calced_Freq  = Omegas.copy()
        self.modes = Modes
    
    def Component_to_normalise(self, rho, phi, iR, iW, mode):
        ci = self.Field_coeffs[iR, iW, mode]
        bi = self.Field_beta  [iR, iW, mode]
        a  = self.Calced_Radii[iR]
        wi = self.Calced_Freq[iW]
        Ez, Hz, Ep, Hp, Er, Hr = Field(rho,phi,ci[0], ci[1], ci[2], ci[3], bi, a, self.n1(wi), self.n2(wi), wi / c, 1)
        return 2 * np.real((Er * np.conj(Hp) - np.conj(Hr) * Ep))
    
    def Normalisation_constants(self, eps = 1e-5):
        
        nr = len(self.Calced_Radii)
        nw = len(self.Calced_Freq)
        nm = len(self.modes)
        N = np.zeros((nr, nw, nm))
        Error = np.zeros((nr, nw, nm,2))
        for ia, a in enumerate(self.Calced_Radii):
            for iW, W in enumerate(self.Calced_Freq):
                for iM,M in enumerate(self.modes):
                    k0 = W / c
                    bi = self.Field_beta[ia, iW, iM]
                    ci = self.Field_coeffs[ia,iW,iM]
                    
                    def func(r):
                        _r = np.array([r])
                        Ez,Hz, Ep,Hp,Er, Hr = Field(_r,0.0,ci[0], ci[1], 
                                                    ci[2], ci[3], bi, a, 
                                                    self.n1(W), self.n2(W), 
                                                    k0, M)
                        
                        return (2 * np.real(Er * np.conj(Hp))  - 2 * np.real(np.conj(Hr) * Ep)) * r
                    
                    res1 = quad(func, 0+eps, a-eps,  epsrel = 1e-12, epsabs = 1e-11)
                    res2 = quad(func, a+eps, np.inf, epsrel = 1e-12, epsabs = 1e-11)
                    
                    #print(res1,res2)
                    N[ia,iW, iM] = 2 * np.pi * (res1[0] + res2[0])
                    Error[ia,iW, iM,:] = res1[1], res2[1]
            print('at Radius number : ',ia)
        
        self.Normalisation_consts = N
        self.Integration_error    = Error
        self.Normalised_Field_coeffs = self.Field_coeffs / np.sqrt(N[:,:,:,np.newaxis])
    
    def calculate_field_norm(self,ci, bi,W,a,M,eps = 1e-5):
        k0 = W / c
        def func(r):
            _r = np.array([r])
            Ez,Hz, Ep,Hp,Er, Hr = Field(_r,0.0,ci[0], ci[1], 
                                        ci[2], ci[3], bi, a, 
                                        self.n1(W), self.n2(W), 
                                        k0, M)
            
            return (2 * np.real(Er * np.conj(Hp))  - 2 * np.real(np.conj(Hr) * Ep)) * r
        res1 = quad(func, 0+eps, a-eps,  epsrel = 1e-12)
        res2 = quad(func, a+eps, np.inf, epsrel = 1e-12)
        return res1[0]+ res2[0]
    
    def EffectiveAreas(self,eps = 1e-5, inner_only = False):
        nr = len(self.Calced_Radii)
        nw = len(self.Calced_Freq)
        nm = len(self.modes)
        A_eff1 = np.zeros((nr, nw, nm))
        A_eff2 = np.zeros((nr, nw, nm))
        
        Error = np.zeros((nr, nw, nm,2))
        for ia, a in enumerate(self.Calced_Radii):
            for iW, W in enumerate(self.Calced_Freq):
                for iM,M in enumerate(self.modes):
                    n1 = self.n1(W)
                    n2 = self.n2(W)
                    k0 = W / c
                    bi = self.Field_beta[ia, iW, iM]
                    ci = self.Normalised_Field_coeffs[ia,iW,iM]
                    
                    def func1(r):
                        _r = np.array([r])
                        Ez,Hz, Ep,Hp,Er, Hr = Field(_r,0.0,ci[0], ci[1], 
                                                    ci[2], ci[3], bi, a, 
                                                    self.n1(W), self.n2(W), 
                                                    k0, M)
                        Ef = np.array([Er[0], Ep[0], Ez[0]])
                        Hf = np.array([Hr[0], Hp[0], Hz[0]])
                        Ft = np.cross(Ef, np.conj(Hf))
                        Ft = np.real(Ft)
                        return (Ft**2).sum() * r
                    
                    def func2(r):
                        _r = np.array([r])
                        Ez,Hz, Ep,Hp,Er, Hr = Field(_r,0.0,ci[0], ci[1], 
                                                    ci[2], ci[3], bi, a, 
                                                    self.n1(W), self.n2(W), 
                                                    k0, M)
                        Ef = np.array([Er[0], Ep[0], Ez[0]])
                        return (np.linalg.norm(np.conj(Ef)))**4 * r
                    
                    def func3(r):
                        _r = np.array([r])
                        Ez,Hz, Ep,Hp,Er, Hr = Field(_r,0.0,ci[0], ci[1], 
                                                    ci[2], ci[3], bi, a, 
                                                    self.n1(W), self.n2(W), 
                                                    k0, M)
                        Ef = np.array([Er[0], Ep[0], Ez[0]])
                        Efc = np.conj(Ef)
                        return Efc.dot(Efc) * (Ef.dot(Ef)) * r
                    
                    res1 = quad(func1, 0+eps, a-eps,  epsrel = 1e-12,epsabs = 1e-10)
                    res2 = quad(func1, a+eps, np.inf, epsrel = 1e-12,epsabs = 1e-10)
                    
                    res3 = quad(func2, 0+eps, a-eps,  epsrel = 1e-12,epsabs = 1e-10)
                    res4 = quad(func2, a+eps, np.inf, epsrel = 1e-12,epsabs = 1e-10)
                    
                    res5 = quad(func3, 0+eps, a-eps,  epsrel = 1e-12,epsabs = 1e-10)
                    res6 = quad(func3, a+eps, np.inf, epsrel = 1e-12,epsabs = 1e-10)
                    
                    Vo = (res1[0] + res2[0])
                    
                    if inner_only==False:
                        Vu = (res3[0] + res4[0])
                        Vu2= (res5[0] + res6[0])
                    else:
                        Vu = res3[0] 
                        Vu2= res5[0]
                    print('Acc error: ', res1[1]+res2[1]+res3[1]+res4[1]+res5[1]+res6[1])                    
                    n_avg = (n1 + n2)/2
                    tal = mu0 * 2 * np.pi/(n_avg**2 * ep0 )
                    A_eff1[ia, iW, iM] = tal * Vo / Vu
                    A_eff2[ia, iW, iM] = tal * Vo / Vu2
                    
                    
                    
            print('at Radius number : ',ia)
        
        self.A_eff_1 = A_eff1
        self.A_eff_2 = A_eff2
    
    def set_chi3(self, val):
        self.chi3 = val
    
    def make_A_eff_1_func(self):
        A_effs_1 = self.A_eff_1.copy()
        A_effs_1[np.isnan(A_effs_1)] = 10e9
        A_effs_1[A_effs_1<0]         = 10e9
        self.A_eff_1_func  = interp1d(self.Calced_Radii, A_effs_1, axis = 0)
    
    def make_A_eff_2_func(self):
        A_effs_2 = self.A_eff_2.copy()
        A_effs_2[np.isnan(A_effs_2)] = 10e9
        A_effs_2[A_effs_2<0]         = 10e9
        self.A_eff_2_func  = interp1d(self.Calced_Radii, A_effs_2, axis = 0)
    
    def get_beta_func(self,shift_idx, radius_idx):
        betas = self.Field_beta.copy()
        idx = np.where(np.isnan(betas))
        betas[idx] = self.Calced_Freq[idx[1]]/c
        diff_b_dw  = np.zeros((len(betas[:,0,0]), len(betas[0,:,0])), dtype = np.complex128)
        for i in range(1, len(betas[0,:,0])-1):
            diff_b_dw[:,i] = (betas[:,i+1,0]-betas[:,i-1,0])/(self.Calced_Freq[i+1] - self.Calced_Freq[i-1])
        self.diff_b_dw = diff_b_dw
        assert shift_idx!=0
        assert shift_idx!=len(betas[0,:,0])-1
        ###### Another Beta????
        offset  = betas[radius_idx, shift_idx,0]  + diff_b_dw[radius_idx, shift_idx] * (self.Calced_Freq - self.Calced_Freq[shift_idx])
        offset  = offset[np.newaxis,:, np.newaxis]
        _uval   = betas[radius_idx, shift_idx-1:shift_idx+2,0]
        _dw = self.Calced_Freq[1]-self.Calced_Freq[0]
        _beta_2 = (_uval*np.array([1,-2,1])/(_dw**2)).sum()
        self.BETA_2 = _beta_2
        self.BETA_0 = betas[radius_idx, shift_idx,0]
        
        self.W0   = self.Calced_Freq[shift_idx]
        self.dBdw = diff_b_dw[radius_idx, shift_idx]
        self.offset = offset
        self._beta_func = CubicSpline(self.Calced_Radii,betas - offset)
        return CubicSpline(self.Calced_Radii,betas - offset)
    
    def calcphase(self, profile,zmax,dz, shift_idx, radius_idx):
        from scipy.integrate import simpson, cumulative_trapezoid
        z = np.arange(0,zmax,dz)
        integrated_phase = np.zeros((len(z),len(self.Calced_Freq),len(self.modes)), dtype = np.complex128)
        Profile_z = profile(z)
        beta = self.get_beta_func(shift_idx, radius_idx)
        beta_z = beta(Profile_z)
        integrated_phase[1:,:,:] = cumulative_trapezoid(beta_z,z, axis = 0)
        self.Integrated_Phase = CubicSpline(z,integrated_phase, extrapolate=False)
    
    def set_fixed_params(self,omega,nm,chi3, fR, Raman,profile):
        self.chi3 = chi3
        self.fR   = fR
        self.W    = omega
        self.nm   = nm
        self.front_fac = 1j * omega * mu0 * chi3 / nm**2 / np.sqrt(2*np.pi)
        self.profile = profile
        self.Raman = Raman
    
    def Interp_omega(self,Vals, upscaling):
        return np.interp(self.W, self.Calced_Freq*upscaling, Vals, left = np.nan, right = np.nan)
    
    def ode_f(self, z, A, zp = False, raman = True, mode = 0, plot = False,omega_upscaling = 1):
        Rz = self.profile(z)
        A1 = self.Interp_omega(self.A_eff_1_func(Rz)[:,mode], omega_upscaling)
        A2 = self.Interp_omega(self.A_eff_2_func(Rz)[:,mode], omega_upscaling)
        #A2 = A1.copy()
        
        
        Phase_w = np.exp(1j * self.Interp_omega(self.Integrated_Phase(z)[:,mode], omega_upscaling))
        if plot:
            plt.plot(self.W,A1)
            plt.title(r'$A_{eff,1}$')
            plt.ylim([0,5]); plt.show()
            plt.plot(self.W,A2)
            plt.title(r'$A_{eff,2}$')
            plt.ylim([0,100]); plt.show()
            plt.plot(self.W, A.real)
            plt.plot(self.W, A.imag)
            plt.title(r'$A(w)$'); plt.show()
            plt.plot(self.W, Phase_w)
            plt.title('Phase'); plt.show()
        
        return JesperMail2(self.W, self.chi3, self.nm, 
                          A,
                          A1, A1, Phase_w, self.Raman, 
                          self.fR,zp=zp,raman=raman)

def sech(x):
    return 2/(np.exp(x) + np.exp(-x) )


IFT = sft.ifft#np.fft.ifft 
FT  = sft.fft#np.fft.fft

def JesperMail(w, chi3, nm,A_w, A_eff_1_w, A_eff_2_w, phase_w, raman_w, fR, zp = False, raman = True):
    fact = 1j * w * mu0 * chi3 /(4 * nm**2*np.sqrt(2 * np.pi))
   # other_fact = (w[-1]-w[0])/(2 * np.pi)
    dw         = w[1]-w[0] / np.sqrt(2 * np.pi)
    dt         = 2*np.pi/(len(w) * dw) / np.sqrt(2 * np.pi)
    nw = len(w)
    DT = np.complex128
    A_tilde  = A_w/phase_w/np.power(A_eff_1_w, 1/4)
    Ap_tilde = A_w/phase_w/np.power(A_eff_2_w, 1/4)
    
    if zp: gtil  = np.zeros(2*nw, dtype = DT)
    else:  gtil  = np.zeros(nw, dtype = DT)
    gtil[0:nw]   = A_tilde
    gt           = IFT(gtil)*dw
    gt2          = gt * (gt.conj())
    if raman:
        gw2         = FT(gt2)*dt
        gw2[:]     *= ((1-fR) + fR * raman_w)
        gt2         = IFT(gw2)*dw
    
    gt*=gt2
    gtil         =  FT(gt)*dt
    res1         =  gtil[0:nw].copy()*2*phase_w*np.power(A_eff_1_w, -1/4) 
    
    if zp: gtil = np.zeros(2*nw, dtype = DT)
    else : gtil = np.zeros(nw,   dtype = DT)
    
    gtil[0:nw]   =  Ap_tilde
    gt           =  IFT(gtil)*dw
    gt2          =  gt * (gt.conj()) * (1-fR)
    gt          *=  gt2
    gtil         =  FT(gt)*dt
    res2         =  gtil[0:nw].copy()*phase_w*np.power(A_eff_2_w, -1/4) 
    
    return fact*(res1+res2)#*(other_fact**2)

def JesperMail2(w, chi3, nm,A_w, A_eff_1_w, A_eff_2_w, phase_w, raman_w, fR, zp = False, raman = True):
    fact = 1j * w * mu0 * chi3 /(4 * nm**2*np.sqrt(2 * np.pi))
    other_fact = (w[-1]-w[0])/(2 * np.pi)
    #dw         = w[1]-w[0] / np.sqrt(2 * np.pi)
    #dt         = 2*np.pi/(len(w) * dw) / np.sqrt(2 * np.pi)
    nw = len(w)
    DT = np.complex128
    A_tilde  = A_w/phase_w/np.power(A_eff_1_w, 1/4)
    Ap_tilde = A_w/phase_w/np.power(A_eff_2_w, 1/4)
    
    if zp: gtil  = np.zeros(2*nw, dtype = DT)
    else:  gtil  = np.zeros(nw, dtype = DT)
    gtil[0:nw]   = A_tilde
    gt           = IFT(gtil)
    gt2          = gt * (gt.conj())
    
    if raman:
        gw2         = FT(gt2)
        gw2[:]     *= ((1-fR) + fR * raman_w)
        gt2         = IFT(gw2)
    
    gt*=gt2
    gtil         =  FT(gt)
    res1         =  2*gtil[0:nw].copy()*phase_w*np.power(A_eff_1_w, -1/4) 
    
    if zp: gtil = np.zeros(2*nw, dtype = DT)
    else:  gtil = np.zeros(nw,   dtype = DT)
    
    gtil[0:nw]   =  Ap_tilde
    gt           =  IFT(gtil)
    gt2          =  gt * (gt.conj()) * (1-fR)
    gt          *=  gt2
    gtil         =  FT(gt)
    res2         =  gtil[0:nw].copy()*phase_w*np.power(A_eff_2_w, -1/4) 
    
    return fact*(res1+res2)*(other_fact**2)

tau1 = 12.2 * 10**-9
tau2 = 32.0 * 10**-9

def R(t):
    res = np.exp(-t / tau2) * np.sin(t/tau1) * (tau1**2  + tau2**2)/(tau1*tau2**2)
    res[t<0] = 0
    return res

