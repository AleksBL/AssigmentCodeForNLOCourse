import numpy as np
from OpticalFiber import eq_det, eq_system, null2, VarFiber, Field
from OpticalFiber import R as Raman_t
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.constants import c, micro
from scipy.integrate import solve_ivp


_k0 = 2 * np.pi / (1)
_n1 = 1.45
_n2 = 1.44
M = 1
vals = []
Vi = np.linspace(0.6, 8, 200)
b0 = _k0 * _n2 * 1.00000001
vals += [b0]
xl = []
COEFFS = []

# for V in Vi:
#     a = V / (_k0 * np.sqrt(_n1 ** 2 - _n2 ** 2))

#     def f(beta):
#         return eq_det(beta, a, _n1, _n2, _k0, M)[0]

#     x = fsolve(f, np.array([b0]))
#     if _n2 * _k0 < x < _n1 * _k0:
#         vals += [fsolve(f, np.array([b0]))]
#         b0 = vals[-1]
#         xl += [V]
#         N = null2(eq_system(b0, a, _n1, _n2, _k0, M)[0])  # , rtol = 1e-10)
#         COEFFS += [N]
# plt.plot(xl, np.array(vals[1:])/_k0)

B1 = 0  # 0.69616630
B2 = 0  # 0.4079426
B3 = 0  # 0.8974794
dn = 0.003
pot = 10 ** 9
w1 = 27.556097 * pot
w2 = 16.215871 * pot
w3 = 0.190473416 * pot


def sellmeier(w):
    return (
        1.44 ** 2
        + B1 / (1 - (w / w1) ** 2)
        + B2 / (1 - (w / w2) ** 2)
        + B3 / (1 - (w / w3) ** 2)
    )


def n1(w):
    return np.sqrt(sellmeier(w)) + dn


def n2(w):
    return 1.44  # np.sqrt(sellmeier(w) )


def cart2pol(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(y, x)
    return r, t


F = VarFiber(n1, n2)



Rmin, Rmax = 6.0, 7.0
wmin, wmax =1.5, 6.0
nR        = 40
nW        = 201
prof_osci = 1*10**6
osci_amp  = 0.99
zmax      = 8*16*10**5  # micrometer



Omegas = c * np.linspace(wmin, wmax, nW)
plt.plot(Omegas, n1(Omegas))
R = np.linspace(Rmin, Rmax, nR)
F.find_modes(R, Omegas, [1], mode="1")

# In[]
plt.show()
plt.plot(Omegas/c, F.Field_beta[:, :, 0].T / (Omegas[np.newaxis, :].T / c))
plt.ylabel(r'modeindex $\beta/k_0$',size = 16)
plt.xlabel(r'$\omega$ [$10^{15}$ Hz]',size = 16)
plt.legend([r"$m_l = 1$," + " R = " + str(R[i])+'$\mu$m' for i in range(nR)])
plt.title( r'$n_1 = 1.443$, $n_2 = 1.440$')
plt.tight_layout()
plt.savefig('Modeindexplot.pdf',dpi = 100)

plt.show()
plt.pause(0.5)

# assert 1 == 0

# In[]

iW =90
iR = 2

plt.legend()
x, y = np.meshgrid(
    np.linspace(-1.2 * Rmax, 1.2 * Rmax, 1000),
    np.linspace(-1.2 * Rmax, 1.2 * Rmax, 1000),
    indexing="ij",
)

rho = np.linspace(0, 1.2 * Rmax, 10000)
r, t = cart2pol(x, y)

b = F.Field_beta
bi = b[iR, iW, 0]
ci = F.Field_coeffs[iR, iW, 0]
wi = Omegas[iW]
a  = R[iR]

Ez1, Hz1, Ep1, Hp1, Er1, Hr1 = Field(
    r, t, ci[0], ci[1], ci[2], ci[3], bi, a, n1(wi), n2(wi), wi / c, 1
)
Ez2, Hz2, Ep2, Hp2, Er2, Hr2 = Field(
    rho, 0.0, ci[0], ci[1], ci[2], ci[3], bi, a, n1(wi), n2(wi), wi / c, 1
)

plt.imshow(np.real(Ez1))
plt.xticks([])
plt.yticks([])
plt.title(r'Re[$E_z(\vec{\rho})$]')
plt.savefig('Re[Ez]_full.pdf',dpi = 100)
plt.show()



tz,tr,tp = 1.0,0.04, 0.05
plt.plot(rho, np.abs(Ez2)*tz,label = str(tz)+r'$E_z$')
plt.plot(rho, np.abs(Er2)*tr,label = str(tr)+r'$E_r$')
plt.plot(rho, np.abs(Ep2)*tp,label = str(tp)+r'$E_\phi$')
plt.xlabel(r'$\rho$ [$\mu$m]',size = 16)
plt.ylabel(r'Electric field [$V/\mu$m]',size = 16)
plt.legend()
plt.title('E-Field components for $R=6.5\mu$m')
plt.tight_layout()
plt.savefig('Ezrp_rho.pdf',dpi = 100)
plt.show()

plt.plot(rho, np.abs(Er2),label = r'$E_r$');
plt.plot(rho, np.abs(Ep2),label = r'$E_\phi$');
plt.xlim([6.4,6.6]); 
plt.ylim([1.15, 1.25])
plt.legend()
plt.xlabel(r'$\rho$ [$\mu$m]',size = 16)
plt.ylabel(r'Electric field [$V/\mu$m]',size = 16)
plt.title('E-Field discontinuity for $R=6.5\mu$m')
plt.tight_layout()
plt.savefig('discontinuity.pdf',dpi = 100)


# In[]
F.Normalisation_constants(eps=1e-5)
F.EffectiveAreas(inner_only=True)


# In[]
plt.plot(F.Calced_Freq/c, F.A_eff_1[:,:,0].T)
plt.ylim([0,5])
plt.xlabel(r'$\omega$ [$10^{15}Hz$]',size = 16)
plt.ylabel(r'$\mathcal{A}_{eff}$ [$\mu$m$^2$]',size = 16)
plt.legend([r" R = " + str(R[i])+'$\mu$m' for i in range(nR)])
plt.title(r'$\mathcal{A}$',size = 16)
plt.savefig('Aeff.pdf', dpi = 100)
plt.show()

plt.plot(F.Calced_Freq/c, F.A_eff_2[:,:,0].T)
plt.ylim([0,0.8e+7])
plt.xlabel(r'$\omega$ [$10^{15}Hz$]',size = 16)
plt.ylabel(r'$\mathcal{B}_{eff}$ [$\mu$m$^2$]',size = 16)
plt.legend([r" R = " + str(R[i])+'$\mu$m' for i in range(nR)])
plt.savefig('Beff.pdf', dpi = 100)
plt.show()


# In[]
F.make_A_eff_1_func()
F.make_A_eff_2_func()
# In[]
# zmax*=8



def Profile(z):
    return (Rmin + Rmax) / 2 +osci_amp* (Rmax - Rmin) / 2 * np.sin(
        2 * np.pi * z / prof_osci
    )


F.calcphase(Profile, zmax, zmax / 100001,  nW//2,  nR//2)

b = F.Field_beta
bi = b[iR, iW, 0]
ci = F.Field_coeffs[iR, iW, 0]
wi = Omegas[iW]
a = R[iR]

Ez2, Hz2, Ep2, Hp2, Er2, Hr2 = Field(
    rho, 0.0, ci[0], ci[1], ci[2], ci[3], bi, a, n1(wi), n2(wi), wi / c, 1
)

plt.plot(rho, np.abs(Ep2))
plt.show()

n_omega =1001
sampling_omega = np.linspace(wmin , wmax, n_omega) * c
dw = sampling_omega[1] - sampling_omega[0]
dt = 1/ (dw * n_omega)
times = np.fft.fftshift(np.fft.fftfreq(2 * n_omega - 1, dw))
Rw = np.fft.rfft(Raman_t(times))*dt/np.sqrt(2*np.pi)
Rw /= np.trapz(Raman_t(times),dx = dt)
#Rw[:] = 0

plt.plot(sampling_omega, np.abs(Rw))
plt.title("Raman")
plt.show()

# In[]
chi3 = 1.61e-10
fR   = 0.0
F.set_fixed_params(sampling_omega.copy(), 1.44, chi3, fR, Rw, Profile)


ts = 100*10**-15*10**6
ws = 2/np.pi/ts

from scipy.constants import mu_0
gamma = 2 * F.W0 * mu_0 * chi3 / (4 * F.A_eff_1_func(Rmax)[nW//2,0] * 1.44**2)
Psol = 1*abs(F.BETA_2)/(ts**2 * gamma)

def sech(x):
    return 2 / (np.exp(-x) + np.exp(x))

def inc_pulse_w():
    x = np.linspace(-dw*n_omega/2,dw*n_omega/2, n_omega)/ws
    return 2*np.sqrt(Psol)/ws*sech(x)

A0 = 5.0 * inc_pulse_w().astype(np.complex128)

#A0 =50000* ts/2/np.pi**0.5 * np.exp(-(10**6*np.linspace(-dw*n_omega/2,dw*n_omega/2, n_omega))**2 / (2/ts)**2)
plt.plot(A0)
plt.show()

# In[]
div = 1
res = solve_ivp(
                F.ode_f,
                (0, zmax/div),
                A0,
                method="RK45",
                t_eval=np.linspace(0, zmax/div, n_omega//2),
                rtol = 1e-15,
                atol = 1e-15
)

PN = np.sum(np.abs(res.y)**2/(sampling_omega[:,np.newaxis]),axis = 0)
print(res.message)
print(res.nfev)

plt.imshow(np.abs(res.y)); plt.colorbar()
