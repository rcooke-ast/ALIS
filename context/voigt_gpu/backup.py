import math
import numpy as np
#from numba import cuda
#from numba import types
from scipy.special import wofz
from matplotlib import pyplot as plt

def voigt(wave, p0, p1, p2, lam, fvl, gam):
    cold = 10.0**p0
    zp1=p1+1.0
    wv=lam*1.0e-8
    bl=p2*wv/2.99792458E5
    a=gam*wv*wv/(3.76730313461770655E11*bl)
    cns=wv*wv*fvl/(bl*2.002134602291006E12)
    cne=cold*cns
    ww=(wave*1.0e-8)/zp1
    v=wv*ww*((1.0/ww)-(1.0/wv))/bl
    print(a, v)
    tau = cne*wofz(v + 1j * a).real
    return np.exp(-1.0*tau)

#@cuda.jit
def voigt_gpu(wave, p0, p1, p2, lam, fvl, gam, flux):
    idx = cuda.grid(1)
    cold = 10.0**p0
    zp1=p1+1.0
    wv=lam*1.0e-8
    bl=p2*wv/2.99792458E5
    a=gam*wv*wv/(3.76730313461770655E11*bl)
    cns=wv*wv*fvl/(bl*2.002134602291006E12)
    cne=cold*cns
    ww=(wave[idx]*1.0e-8)/zp1
    v=wv*ww*((1.0/ww)-(1.0/wv))/bl
    z = types.complex128(v + 1j * a)
    tau = cne*math.exp(-(v*v-a*a))*math.cos(2.0*v*a)
#    tau = cne*math.exp(-z*z)#(math.exp(-z*z)*math.erfc(-1j*z)).real
    flux[idx] = math.exp(-1.0*tau)

wave = np.arange(1200.0, 1230.0, 0.001, dtype=np.float32)
lam, fvl, gam = 1215.6701, 0.4164, 6.265E8
N, z, b = 20.0, 0.0, 10.0

# Run on CPU
flux_cpu = voigt(wave, N, z, b, lam, fvl, gam)

plt.plot(wave, flux_cpu, 'k-', drawstyle='steps-mid')
plt.show()

# Run on GPU
# Setup variables for the GPU
#d_wave = cuda.to_device(wave)
#flux_gpu = cuda.to_device(np.zeros(shape=d_wave.shape, dtype=np.float32))
#blocks, threads_per_block = 1 + wave.size//128, 128
#voigt_gpu[blocks, threads_per_block](d_wave, N, z, b, lam, fvl, gam, flux_gpu)

# Check they are equal
#np.testing.assert_almost_equal(flux_cpu, flux_gpu.copy_to_host(), decimal=5)
