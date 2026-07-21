from math import cos, sin, floor, exp, copysign, nan, inf, isnan, isinf
import math
import timeit
import numpy
from numba import cuda
from numba import types
from scipy.special import wofz
from matplotlib import pyplot as plt

@cuda.jit(device=True)
def erfcx_y100(y100, erfcx_cc):
    iy100 = int(y100)
    if iy100 == 100:
        return 1.0
    else:
        t = 2.0*y100 - (2*types.float32(iy100) + 1.0)
        return erfcx_cc[iy100, 0] + (erfcx_cc[iy100, 1] + (erfcx_cc[iy100, 2] + (erfcx_cc[iy100, 3] + (erfcx_cc[iy100, 4] + (erfcx_cc[iy100, 5] + erfcx_cc[iy100, 6] * t) * t) * t) * t) * t) * t

@cuda.jit(device=True)
def sincomplex(x, sinx):
    if abs(x) < 1e-4:
        return 1 - (0.1666666666666666666667)*x*x
    else:
        return sinx/x

@cuda.jit(device=True)
def sinh_taylor(x):
    return x * (1 + (x*x) * (0.1666666666666666666667 + 0.00833333333333333333333 * (x*x)))

@cuda.jit(device=True)
def sqr(x):
    return x*x

@cuda.jit(device=True)
def faddeeva_re(x, erfcx_cc):
    if (x >= 0):
        if (x > 50):  # continued-fraction expansion is faster
            ispi = 0.56418958354775628694807945156  # 1 / sqrt(pi)
            if (x > 5e7):  # 1-term expansion, important to avoid overflow
                return ispi / x
            return ispi*((x*x) * (x*x+4.5) + 2) / (x * ((x*x) * (x*x+5) + 3.75))
        return erfcx_y100(400.0/(4.0+x), erfcx_cc)
    else:
        if x < -26.7:
            return inf
        else:
            if x < -6.1:
                return 2*exp(x*x)
            else:
                return 2*exp(x*x) - erfcx_y100(400.0/(4.0-x), erfcx_cc)

@cuda.jit(device=True)
def faddeeva_real(z, erfcx_cc, expa2n2):
    relerr = 1.0E-7
    a = 0.518321480430085929872   # pi / sqrt(-log(eps*0.5))
    c = 0.329973702884629072537   # (2/pi) * a;
    a2 = 0.268657157075235951582  # a^2
    x = abs(z.real)
    y = z.imag
    ya = abs(y)

    sum1, sum2, sum3, sum4, sum5 = 0, 0, 0, 0, 0

    if (ya > 7 or (x > 6 and (ya > 0.1 or (x > 8 and ya > 1e-10) or x > 28))):
        ispi = 0.56418958354775628694807945156  # 1 / sqrt(pi)
        if y < 0:
            xs = -z.real
        else:
            xs = z.real
        if (x + ya > 4000):  # nu <= 2
            if (x + ya > 1e7):  # nu == 1, w(z) = i/sqrt(pi) / z
                if (x > ya):
                    yax = ya / xs
                    denom = ispi / (xs + yax*ya)
                    ret = denom*yax
                elif isinf(ya):
                    if isnan(x) or y < 0:
                        return nan
                    else:
                        return 0
                else:
                    xya = xs / ya
                    denom = ispi / (xya*xs + ya)
                    ret = denom
            else:  # nu == 2, w(z) = i/sqrt(pi) * z / (z*z - 0.5)
                dr = xs*xs - ya*ya - 0.5
                di = 2*xs*ya
                denom = ispi / (dr*dr + di*di)
                ret = denom * (xs*di-ya*dr)
        else:  # compute nu(z) estimate and do general continued fraction
            c0, c1, c2, c3, c4 = 3.9, 11.398, 0.08254, 0.1421, 0.2023  # fit
            nu = floor(c0 + c1 / (c2*x + c3*ya + c4))
            wr = xs
            wi = ya
            nu = 0.5 * (nu - 1)
            while nu > 0.4:
#            for (nu = 0.5 * (nu - 1); nu > 0.4; nu -= 0.5):
                denom = nu / (wr*wr + wi*wi)
                wr = xs - wr * denom
                wi = ya + wi * denom
                nu -= 0.5
            """
            { // w(z) = i/sqrt(pi) / w:
                denom = ispi / (wr*wr + wi*wi)
                ret = complex(denom*wi, denom*wr)
            }
            """
            denom = ispi / (wr*wr + wi*wi)
            ret = denom*wi
        if (y < 0):
            val = 2.0*exp((ya-xs)*(xs+ya))
            if val == 0.0:
                return -ret
            else:
                return val*cos(2*xs*y) - ret
        else:
            return ret
    elif (x < 10):
        prod2ax, prodm2ax = 1.0, 1.0
        if (isnan(y)):
            return y

        if (x < 5e-4):
            x2 = x*x
            expx2 = 1 - x2 * (1 - 0.5*x2)  # exp(-x*x) via Taylor
            # compute exp(2*a*x) and exp(-2*a*x) via Taylor, to double precision
            ax2 = 1.036642960860171859744*x  # 2*a*x
            exp2ax = 1 + ax2 * (1 + ax2 * (0.5 + 0.166666666666666666667*ax2))
            expm2ax = 1 - ax2 * (1 - ax2 * (0.5 - 0.166666666666666666667*ax2))
            n = 1
            while True:
#                for (int n = 1; 1; ++n) {
                coef = expa2n2[n-1] * expx2 / (a2*(n*n) + y*y)
                prod2ax *= exp2ax
                prodm2ax *= expm2ax
                sum1 += coef
                sum2 += coef * prodm2ax
                sum3 += coef * prod2ax
                
                # really = sum5 - sum4
                sum5 += coef * (2*a) * n * sinh_taylor((2*a)*n*x)
                
                # test convergence via sum3
                if (coef * prod2ax < relerr * sum3):
                    break
                n += 1
        else:  # x > 5e-4, compute sum4 and sum5 separately
            expx2 = exp(-x*x)
            exp2ax = exp((2*a)*x)
            expm2ax = 1 / exp2ax
            n = 1
            while True:
#                for (int n = 1; 1; ++n) {
                coef = expa2n2[n-1] * expx2 / (a2*(n*n) + y*y)
                prod2ax *= exp2ax
                prodm2ax *= expm2ax
                sum1 += coef
                sum2 += coef * prodm2ax
                sum4 += (coef * prodm2ax) * (a*n)
                sum3 += coef * prod2ax
                sum5 += (coef * prod2ax) * (a*n)
                # test convergence via sum5, since this sum has the slowest decay
                if ((coef * prod2ax) * (a*n) < relerr * sum5):
                    break
                n += 1
        if y > -6:
            expx2erfcxy = expx2*faddeeva_re(y, erfcx_cc)
        else:
            expx2erfcxy = 2*exp(y*y-x*x)
        if (y > 5):  # imaginary terms cancel
            sinxy = sin(x*y)
            ret = (expx2erfcxy - c*y*sum1) * cos(2*x*y) + (c*x*expx2) * sinxy * sincomplex(x*y, sinxy)
        else:
            xs = z.real
            sinxy = sin(xs*y)
            sin2xy = sin(2*xs*y)
            cos2xy = cos(2*xs*y)
            coef1 = expx2erfcxy - c*y*sum1
            coef2 = c*xs*expx2
            ret = coef1 * cos2xy + coef2 * sinxy * sincomplex(xs*y, sinxy)
    else:  #x large: only sum3 & sum5 contribute (see above note)
        if (isnan(x)):
            return x
        if (isnan(y)):
            return y

        ret = exp(-x*x)  # |y| < 1e-10, so we only need exp(-x*x) term
        # (round instead of ceil as in original paper; note that x/a > 1 here)
        n0 = floor(x/a + 0.5)  # sum in both directions, starting at n0
        dx = a*n0 - x
        sum3 = exp(-dx*dx) / (a2*(n0*n0) + y*y)
        sum5 = a*n0 * sum3
        exp1 = exp(4*a*dx)
        exp1dn = 1
        dn = 1
        while n0 - dn > 0:
#        for (dn = 1; n0 - dn > 0; ++dn):  # loop over n0-dn and n0+dn terms
            np = n0 + dn
            nm = n0 - dn
            tp = exp(-sqr(a*dn+dx))
            exp1dn *= exp1
            tm = tp * exp1dn  # trick to get tm from tp
            tp /= (a2*(np*np) + y*y)
            tm /= (a2*(nm*nm) + y*y)
            sum3 += tp + tm
            sum5 += a * (np * tp + nm * tm)
            if (a * (np * tp + nm * tm) < relerr * sum5):
                return ret + (0.5*c)*y*(sum2+sum3)
            dn += 1
        while True:  # loop over n0+dn terms only (since n0-dn <= 0)
            np = n0 + (dn+1)
            tp = exp(-sqr(a*dn+dx)) / (a2*(np*np) + y*y)
            sum3 += tp
            sum5 += a * np * tp
            if (a * np * tp < relerr * sum5):
                return ret + (0.5*c)*y*(sum2+sum3)
    return ret + (0.5*c)*y*(sum2+sum3)

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
    tau = cne*wofz(v + 1j * a).real
    return numpy.exp(-1.0*tau)

@cuda.jit
def voigt_gpu(wave, p0, p1, p2, lam, fvl, gam, erfcx_cc, expa2n2, flux):
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
    tau = cne*faddeeva_real(z, erfcx_cc, expa2n2)
    flux[idx] = math.exp(-1.0*tau)

# Setup the shared memory data
expa2n2 = numpy.array([7.64405281671221563e-01, 3.41424527166548425e-01, 8.91072646929412548e-02, 1.35887299055460086e-02, 1.21085455253437481e-03, 6.30452613933449404e-05, 1.91805156577114683e-06, 3.40969447714832381e-08, 3.54175089099469393e-10, 2.14965079583260682e-12, 7.62368911833724354e-15, 1.57982797110681093e-17, 1.91294189103582677e-20, 1.35344656764205340e-23, 5.59535712428588720e-27, 1.35164257972401769e-30, 1.90784582843501167e-34, 1.57351920291442930e-38, 7.58312432328032845e-43, 2.13536275438697082e-47, 3.51352063787195769e-52, 3.37800830266396920e-57, 1.89769439468301000e-62, 6.22929926072668851e-68, 1.19481172006938722e-73, 1.33908181133005953e-79, 8.76924303483223939e-86, 3.35555576166254986e-92, 7.50264110688173024e-99, 9.80192200745410268e-106, 7.48265412822268959e-113, 3.33770122566809425e-120, 8.69934598159861140e-128, 1.32486951484088852e-135, 1.17898144201315253e-143, 6.13039120236180012e-152, 1.86258785950822098e-160, 3.30668408201432783e-169, 3.43017280887946235e-178, 2.07915397775808219e-187, 7.36384545323984966e-197, 1.52394760394085741e-206, 1.84281935046532100e-216, 1.30209553802992923e-226, 5.37588903521080531e-237, 1.29689584599763145e-247, 1.82813078022866562e-258, 1.50576355348684241e-269, 7.24692320799294194e-281, 2.03797051314726829e-292, 3.34880215927873807e-304, 0.0], dtype=numpy.float64)

erfcx_cc = numpy.loadtxt("erfcx_coeffs.dat", delimiter=',').astype(numpy.float64)

wave = numpy.arange(1200.0, 1230.0, 0.001, dtype=numpy.float64)
lam, fvl, gam = 1215.6701, 0.4164, 6.265E8
N, z, b = 20.0, 0.0, 10.0

# Run on CPU
flux_cpu = voigt(wave, N, z, b, lam, fvl, gam)

#plt.plot(wave, flux_cpu, 'k-', drawstyle='steps-mid')
#plt.show()

# Run on GPU
# Setup variables for the GPU
d_wave = cuda.to_device(wave)
d_erfcx_cc = cuda.to_device(erfcx_cc)
d_expa2n2 = cuda.to_device(expa2n2)
flux_gpu = cuda.to_device(numpy.zeros(shape=d_wave.shape, dtype=numpy.float64))
blocks, threads_per_block = 1 + wave.size//128, 128
voigt_gpu[blocks, threads_per_block](d_wave, N, z, b, lam, fvl, gam, d_erfcx_cc, d_expa2n2, flux_gpu)

print(np.min(d_wave))
print(np.max(d_wave))

print(timeit.timeit('voigt(wave, N, z, b, lam, fvl, gam)', globals=globals(), number=1000))
print(timeit.timeit('voigt_gpu[blocks, threads_per_block](d_wave, N, z, b, lam, fvl, gam, d_erfcx_cc, d_expa2n2, flux_gpu)', globals=globals(), number=1000))

# Check they are equal
gpuflux = flux_gpu.copy_to_host()
numpy.testing.assert_almost_equal(flux_cpu, gpuflux, decimal=5)

print(numpy.max(numpy.abs(gpuflux-flux_cpu)))
#print(gpuflux)
#print(flux_cpu)

plt.plot(wave, gpuflux, 'k-', drawstyle='steps-mid')
plt.plot(wave, flux_cpu, 'r--', drawstyle='steps-mid')
plt.show()

