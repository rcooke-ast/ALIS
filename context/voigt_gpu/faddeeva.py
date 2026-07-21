from math import cos, sin, floor, exp, copysign, nan, inf, isnan, isinf
from cmath import exp as cexp
import timeit
import numpy

DBL_EPSILON = numpy.finfo(float).eps
erfcx_cc = numpy.loadtxt("erfcx_coeffs.dat", delimiter=',')

def cpolar(r, t):
    if r == 0.0:# and !isnan(t))
        return 0.0
    else:
        return complex(r * cos(t), r * sin(t))

def sincomplex(x, sinx):
    if abs(x) < 1e-4:
        return 1 - (0.1666666666666666666667)*x*x
    else:
        return sinx/x

def sinh_taylor(x):
    return x * (1 + (x*x) * (0.1666666666666666666667 + 0.00833333333333333333333 * (x*x)))

def sqr(x):
    return x*x

def relerr(a, b):
    if (isnan(a) or isnan(b) or isinf(a) or isinf(b)):
        if ((isnan(a) and not isnan(b)) or (not isnan(a) and isnan(b)) or
                (isinf(a) and not isinf(b)) or (not isinf(a) and isinf(b)) or
                (isinf(a) and isinf(b) and a*b < 0)):
            return inf
        return 0
    if (a == 0):
        if b == 0:
            return 0
        else:
            return inf
    else:
        return abs((b-a) / a)

expa2n2 = [7.64405281671221563e-01, 3.41424527166548425e-01, 8.91072646929412548e-02, 1.35887299055460086e-02, 1.21085455253437481e-03, 6.30452613933449404e-05, 1.91805156577114683e-06, 3.40969447714832381e-08, 3.54175089099469393e-10, 2.14965079583260682e-12, 7.62368911833724354e-15, 1.57982797110681093e-17, 1.91294189103582677e-20, 1.35344656764205340e-23, 5.59535712428588720e-27, 1.35164257972401769e-30, 1.90784582843501167e-34, 1.57351920291442930e-38, 7.58312432328032845e-43, 2.13536275438697082e-47, 3.51352063787195769e-52, 3.37800830266396920e-57, 1.89769439468301000e-62, 6.22929926072668851e-68, 1.19481172006938722e-73, 1.33908181133005953e-79, 8.76924303483223939e-86, 3.35555576166254986e-92, 7.50264110688173024e-99, 9.80192200745410268e-106, 7.48265412822268959e-113, 3.33770122566809425e-120, 8.69934598159861140e-128, 1.32486951484088852e-135, 1.17898144201315253e-143, 6.13039120236180012e-152, 1.86258785950822098e-160, 3.30668408201432783e-169, 3.43017280887946235e-178, 2.07915397775808219e-187, 7.36384545323984966e-197, 1.52394760394085741e-206, 1.84281935046532100e-216, 1.30209553802992923e-226, 5.37588903521080531e-237, 1.29689584599763145e-247, 1.82813078022866562e-258, 1.50576355348684241e-269, 7.24692320799294194e-281, 2.03797051314726829e-292, 3.34880215927873807e-304, 0.0]

def faddeeva(z):
    relerr = DBL_EPSILON
    a = 0.518321480430085929872   # pi / sqrt(-log(eps*0.5))
    c = 0.329973702884629072537   # (2/pi) * a;
    a2 = 0.268657157075235951582  # a^2
    x = abs(z.real)
    y = z.imag
    ya = abs(y)

    ret = 0.  # return value

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
                    ret = complex(denom*yax, denom)
                elif isinf(ya):
                    if isnan(x) or y < 0:
                        return complex(nan,nan)
                    else:
                        return complex(0,0)
                else:
                    xya = xs / ya
                    denom = ispi / (xya*xs + ya)
                    ret = complex(denom, denom*xya)
            else:  # nu == 2, w(z) = i/sqrt(pi) * z / (z*z - 0.5)
                dr = xs*xs - ya*ya - 0.5
                di = 2*xs*ya
                denom = ispi / (dr*dr + di*di)
                ret = complex(denom * (xs*di-ya*dr), denom * (xs*dr+ya*di))
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
            ret = complex(denom*wi, denom*wr)
        if (y < 0):
            return 2.0*cexp(complex((ya-xs)*(xs+ya), 2*xs*y)) - ret
        else:
            return ret
    elif (x < 10):
        prod2ax, prodm2ax = 1.0, 1.0
        if (isnan(y)):
            return complex(y,y)

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
            expx2erfcxy = expx2*faddeeva_re(y)
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
            ret = complex(coef1 * cos2xy + coef2 * sinxy * sincomplex(xs*y, sinxy),
                    coef2 * sincomplex(2*xs*y, sin2xy) - coef1 * sin2xy)
    else:  #x large: only sum3 & sum5 contribute (see above note)
        if (isnan(x)):
            return complex(x,x)
        if (isnan(y)):
            return complex(y,y)

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
                return ret + complex((0.5*c)*y*(sum2+sum3), (0.5*c)*copysign(sum5-sum4, z.real))
            dn += 1
        while True:  # loop over n0+dn terms only (since n0-dn <= 0)
            np = n0 + (dn+1)
            tp = exp(-sqr(a*dn+dx)) / (a2*(np*np) + y*y)
            sum3 += tp
            sum5 += a * np * tp
            if (a * np * tp < relerr * sum5):
                return ret + complex((0.5*c)*y*(sum2+sum3), (0.5*c)*copysign(sum5-sum4, z.real))
    return ret + complex((0.5*c)*y*(sum2+sum3), (0.5*c)*copysign(sum5-sum4, z.real))

def faddeeva_real(z):
    relerr = DBL_EPSILON
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
            expx2erfcxy = expx2*faddeeva_re(y)
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

def faddeeva_re(x):
    if (x >= 0):
        if (x > 50):  # continued-fraction expansion is faster
            ispi = 0.56418958354775628694807945156  # 1 / sqrt(pi)
            if (x > 5e7):  # 1-term expansion, important to avoid overflow
                return ispi / x
            return ispi*((x*x) * (x*x+4.5) + 2) / (x * ((x*x) * (x*x+5) + 3.75))
        return erfcx_y100(400.0/(4.0+x))
    else:
        if x < -26.7:
            return HUGE_VAL
        else:
            if x < -6.1:
                return 2*exp(x*x)
            else:
                return 2*exp(x*x) - erfcx_y100(400/(4-x))

def erfcx_y100(y100):
    iy100 = int(y100)
    if iy100 == 100:
        return 1.0
    else:
        t = 2.0*y100 - (2*iy100 + 1.0)
        return erfcx_cc[iy100, 0] + (erfcx_cc[iy100, 1] + (erfcx_cc[iy100, 2] + (erfcx_cc[iy100, 3] + (erfcx_cc[iy100, 4] + (erfcx_cc[iy100, 5] + erfcx_cc[iy100, 6] * t) * t) * t) * t) * t) * t

def test_faddeeva(verbose=False):
        ztst = [
            complex(624.2,-0.26123),
            complex(-0.4,3.),
            complex(0.6,2.),
            complex(-1.,1.),
            complex(-1.,-9.),
            complex(-1.,9.),
            complex(-0.0000000234545,1.1234),
            complex(-3.,5.1),
            complex(-53,30.1),
            complex(0.0,0.12345),
            complex(11,1),
            complex(-22,-2),
            complex(9,-28),
            complex(21,-33),
            complex(1e5,1e5),
            complex(1e14,1e14),
            complex(-3001,-1000),
            complex(1e160,-1e159),
            complex(-6.01,0.01),
            complex(-0.7,-0.7),
            complex(2.611780000000000e+01, 4.540909610972489e+03),
            complex(0.8e7,0.3e7),
            complex(-20,-19.8081),
            complex(1e-16,-1.1e-16),
            complex(2.3e-8,1.3e-8),
            complex(6.3,-1e-13),
            complex(6.3,1e-20),
            complex(1e-20,6.3),
            complex(1e-20,16.3),
            complex(9,1e-300),
            complex(6.01,0.11),
            complex(8.01,1.01e-10),
            complex(28.01,1e-300),
            complex(10.01,1e-200),
            complex(10.01,-1e-200),
            complex(10.01,0.99e-10),
            complex(10.01,-0.99e-10),
            complex(1e-20,7.01),
            complex(-1,7.01),
            complex(5.99,7.01),
            complex(1,0),
            complex(55,0),
            complex(-0.1,0),
            complex(1e-20,0),
            complex(0,5e-14),
            complex(0,51)]#,
#            complex(inf,0),
#            complex(-inf,0),
#            complex(0,inf),
#            complex(0,-inf),
#            complex(inf,inf),
#            complex(inf,-inf),
#            complex(nan,nan),
#            complex(nan,0),
#            complex(0,nan),
#            complex(nan,inf),
#            complex(inf,nan)]
        wans = [
            complex(-3.78270245518980507452677445620103199303131110e-7,
                0.000903861276433172057331093754199933411710053155),
            complex(0.1764906227004816847297495349730234591778719532788,
                -0.02146550539468457616788719893991501311573031095617),
            complex(0.2410250715772692146133539023007113781272362309451,
                0.06087579663428089745895459735240964093522265589350),
            complex(0.30474420525691259245713884106959496013413834051768,
                -0.20821893820283162728743734725471561394145872072738),
            complex(7.317131068972378096865595229600561710140617977e34,
                8.321873499714402777186848353320412813066170427e34),
            complex(0.0615698507236323685519612934241429530190806818395,
                -0.00676005783716575013073036218018565206070072304635),
            complex(0.3960793007699874918961319170187598400134746631,
                -5.593152259116644920546186222529802777409274656e-9),
            complex(0.08217199226739447943295069917990417630675021771804,
                -0.04701291087643609891018366143118110965272615832184),
            complex(0.00457246000350281640952328010227885008541748668738,
                -0.00804900791411691821818731763401840373998654987934),
            complex(0.8746342859608052666092782112565360755791467973338452,
                0.),
            complex(0.00468190164965444174367477874864366058339647648741,
                0.0510735563901306197993676329845149741675029197050),
            complex(-0.0023193175200187620902125853834909543869428763219,
                -0.025460054739731556004902057663500272721780776336),
            complex(9.11463368405637174660562096516414499772662584e304,
                3.97101807145263333769664875189354358563218932e305),
            complex(-4.4927207857715598976165541011143706155432296e281,
                -2.8019591213423077494444700357168707775769028e281),
            complex(2.820947917809305132678577516325951485807107151e-6,
                2.820947917668257736791638444590253942253354058e-6),
            complex(2.82094791773878143474039725787438662716372268e-15,
                2.82094791773878143474039725773333923127678361e-15),
            complex(-0.0000563851289696244350147899376081488003110150498,
                -0.000169211755126812174631861529808288295454992688),
            complex(-5.586035480670854326218608431294778077663867e-162,
                5.586035480670854326218608431294778077663867e-161),
            complex(0.00016318325137140451888255634399123461580248456,
                -0.095232456573009287370728788146686162555021209999),
            complex(0.69504753678406939989115375989939096800793577783885,
                -1.8916411171103639136680830887017670616339912024317),
            complex(0.0001242418269653279656612334210746733213167234822,
                7.145975826320186888508563111992099992116786763e-7),
            complex(2.318587329648353318615800865959225429377529825e-8,
                6.182899545728857485721417893323317843200933380e-8),
            complex(-0.0133426877243506022053521927604277115767311800303,
                -0.0148087097143220769493341484176979826888871576145),
            complex(1.00000000000000012412170838050638522857747934,
                1.12837916709551279389615890312156495593616433e-16),
            complex(0.9999999853310704677583504063775310832036830015,
                2.595272024519678881897196435157270184030360773e-8),
            complex(-1.4731421795638279504242963027196663601154624e-15,
                0.090727659684127365236479098488823462473074709),
            complex(5.79246077884410284575834156425396800754409308e-18,
                0.0907276596841273652364790985059772809093822374),
            complex(0.0884658993528521953466533278764830881245144368,
                1.37088352495749125283269718778582613192166760e-22),
            complex(0.0345480845419190424370085249304184266813447878,
                2.11161102895179044968099038990446187626075258e-23),
            complex(6.63967719958073440070225527042829242391918213e-36,
                0.0630820900592582863713653132559743161572639353),
            complex(0.00179435233208702644891092397579091030658500743634,
                0.0951983814805270647939647438459699953990788064762),
            complex(9.09760377102097999924241322094863528771095448e-13,
                0.0709979210725138550986782242355007611074966717),
            complex(7.2049510279742166460047102593255688682910274423e-304,
                0.0201552956479526953866611812593266285000876784321),
            complex(3.04543604652250734193622967873276113872279682e-44,
                0.0566481651760675042930042117726713294607499165),
            complex(3.04543604652250734193622967873276113872279682e-44,
                0.0566481651760675042930042117726713294607499165),
            complex(0.5659928732065273429286988428080855057102069081e-12,
                0.056648165176067504292998527162143030538756683302),
            complex(-0.56599287320652734292869884280802459698927645e-12,
                0.0566481651760675042929985271621430305387566833029),
            complex(0.0796884251721652215687859778119964009569455462,
                1.11474461817561675017794941973556302717225126e-22),
            complex(0.07817195821247357458545539935996687005781943386550,
                -0.01093913670103576690766705513142246633056714279654),
            complex(0.04670032980990449912809326141164730850466208439937,
                0.03944038961933534137558064191650437353429669886545),
            complex(0.36787944117144232159552377016146086744581113103176,
                0.60715770584139372911503823580074492116122092866515),
            complex(0,
                0.010259688805536830986089913987516716056946786526145),
            complex(0.99004983374916805357390597718003655777207908125383,
                -0.11208866436449538036721343053869621153527769495574),
            complex(0.99999999999999999999999999999999999999990000,
                1.12837916709551257389615890312154517168802603e-20),
            complex(0.999999999999943581041645226871305192054749891144158,
                0),
            complex(0.0110604154853277201542582159216317923453996211744250,
                0)]#,
#            complex(0,0),
#            complex(0,0),
#            complex(0,0),
#            complex(inf,0),
#            complex(0,0),
#            complex(nan,nan),
#            complex(nan,nan),
#            complex(nan,nan),
#            complex(nan,0),
#            complex(nan,nan),
#            complex(nan,nan)]
        errmax = 0
        i = 0
        nfail = 0
        while i < len(ztst):
#        for (int i = 0; i < NTST; ++i) {
            fw = faddeeva(ztst[i])
            re_err = relerr(wans[i].real, fw.real)
            im_err = relerr(wans[i].imag, fw.imag)
            if verbose:
                print("w(%g%+gi) = %g%+gi (vs. %g%+gi), re/im rel. err. = %0.2g/%0.2g)" % (
                         ztst[i].real,ztst[i].imag, fw.real, fw.imag, wans[i].real, wans[i].imag,
                         re_err, im_err))
            if (re_err > errmax):
                errmax = re_err
            if (im_err > errmax):
                errmax = im_err
            if re_err > 1e-13 or im_err > 1e-13:
                nfail += 1
            i += 1
        if verbose:
            if (errmax > 1e-13):
                print("FAILURE -- relative error {0:e}} too large!\n".format(errmax))
            else:
                print("SUCCESS (max relative error = {0:e})\n".format(errmax))
            print("NFAIL =", nfail)

def test_faddeeva(verbose=False):
        ztst = [
            complex(624.2,-0.26123),
            complex(-0.4,3.),
            complex(0.6,2.),
            complex(-1.,1.),
            complex(-1.,-9.),
            complex(-1.,9.),
            complex(-0.0000000234545,1.1234),
            complex(-3.,5.1),
            complex(-53,30.1),
            complex(0.0,0.12345),
            complex(11,1),
            complex(-22,-2),
            complex(9,-28),
            complex(21,-33),
            complex(1e5,1e5),
            complex(1e14,1e14),
            complex(-3001,-1000),
            complex(1e160,-1e159),
            complex(-6.01,0.01),
            complex(-0.7,-0.7),
            complex(2.611780000000000e+01, 4.540909610972489e+03),
            complex(0.8e7,0.3e7),
            complex(-20,-19.8081),
            complex(1e-16,-1.1e-16),
            complex(2.3e-8,1.3e-8),
            complex(6.3,-1e-13),
            complex(6.3,1e-20),
            complex(1e-20,6.3),
            complex(1e-20,16.3),
            complex(9,1e-300),
            complex(6.01,0.11),
            complex(8.01,1.01e-10),
            complex(28.01,1e-300),
            complex(10.01,1e-200),
            complex(10.01,-1e-200),
            complex(10.01,0.99e-10),
            complex(10.01,-0.99e-10),
            complex(1e-20,7.01),
            complex(-1,7.01),
            complex(5.99,7.01),
            complex(1,0),
            complex(55,0),
            complex(-0.1,0),
            complex(1e-20,0),
            complex(0,5e-14),
            complex(0,51)]#,
#            complex(inf,0),
#            complex(-inf,0),
#            complex(0,inf),
#            complex(0,-inf),
#            complex(inf,inf),
#            complex(inf,-inf),
#            complex(nan,nan),
#            complex(nan,0),
#            complex(0,nan),
#            complex(nan,inf),
#            complex(inf,nan)]
        wans = [
            complex(-3.78270245518980507452677445620103199303131110e-7,
                0.000903861276433172057331093754199933411710053155),
            complex(0.1764906227004816847297495349730234591778719532788,
                -0.02146550539468457616788719893991501311573031095617),
            complex(0.2410250715772692146133539023007113781272362309451,
                0.06087579663428089745895459735240964093522265589350),
            complex(0.30474420525691259245713884106959496013413834051768,
                -0.20821893820283162728743734725471561394145872072738),
            complex(7.317131068972378096865595229600561710140617977e34,
                8.321873499714402777186848353320412813066170427e34),
            complex(0.0615698507236323685519612934241429530190806818395,
                -0.00676005783716575013073036218018565206070072304635),
            complex(0.3960793007699874918961319170187598400134746631,
                -5.593152259116644920546186222529802777409274656e-9),
            complex(0.08217199226739447943295069917990417630675021771804,
                -0.04701291087643609891018366143118110965272615832184),
            complex(0.00457246000350281640952328010227885008541748668738,
                -0.00804900791411691821818731763401840373998654987934),
            complex(0.8746342859608052666092782112565360755791467973338452,
                0.),
            complex(0.00468190164965444174367477874864366058339647648741,
                0.0510735563901306197993676329845149741675029197050),
            complex(-0.0023193175200187620902125853834909543869428763219,
                -0.025460054739731556004902057663500272721780776336),
            complex(9.11463368405637174660562096516414499772662584e304,
                3.97101807145263333769664875189354358563218932e305),
            complex(-4.4927207857715598976165541011143706155432296e281,
                -2.8019591213423077494444700357168707775769028e281),
            complex(2.820947917809305132678577516325951485807107151e-6,
                2.820947917668257736791638444590253942253354058e-6),
            complex(2.82094791773878143474039725787438662716372268e-15,
                2.82094791773878143474039725773333923127678361e-15),
            complex(-0.0000563851289696244350147899376081488003110150498,
                -0.000169211755126812174631861529808288295454992688),
            complex(-5.586035480670854326218608431294778077663867e-162,
                5.586035480670854326218608431294778077663867e-161),
            complex(0.00016318325137140451888255634399123461580248456,
                -0.095232456573009287370728788146686162555021209999),
            complex(0.69504753678406939989115375989939096800793577783885,
                -1.8916411171103639136680830887017670616339912024317),
            complex(0.0001242418269653279656612334210746733213167234822,
                7.145975826320186888508563111992099992116786763e-7),
            complex(2.318587329648353318615800865959225429377529825e-8,
                6.182899545728857485721417893323317843200933380e-8),
            complex(-0.0133426877243506022053521927604277115767311800303,
                -0.0148087097143220769493341484176979826888871576145),
            complex(1.00000000000000012412170838050638522857747934,
                1.12837916709551279389615890312156495593616433e-16),
            complex(0.9999999853310704677583504063775310832036830015,
                2.595272024519678881897196435157270184030360773e-8),
            complex(-1.4731421795638279504242963027196663601154624e-15,
                0.090727659684127365236479098488823462473074709),
            complex(5.79246077884410284575834156425396800754409308e-18,
                0.0907276596841273652364790985059772809093822374),
            complex(0.0884658993528521953466533278764830881245144368,
                1.37088352495749125283269718778582613192166760e-22),
            complex(0.0345480845419190424370085249304184266813447878,
                2.11161102895179044968099038990446187626075258e-23),
            complex(6.63967719958073440070225527042829242391918213e-36,
                0.0630820900592582863713653132559743161572639353),
            complex(0.00179435233208702644891092397579091030658500743634,
                0.0951983814805270647939647438459699953990788064762),
            complex(9.09760377102097999924241322094863528771095448e-13,
                0.0709979210725138550986782242355007611074966717),
            complex(7.2049510279742166460047102593255688682910274423e-304,
                0.0201552956479526953866611812593266285000876784321),
            complex(3.04543604652250734193622967873276113872279682e-44,
                0.0566481651760675042930042117726713294607499165),
            complex(3.04543604652250734193622967873276113872279682e-44,
                0.0566481651760675042930042117726713294607499165),
            complex(0.5659928732065273429286988428080855057102069081e-12,
                0.056648165176067504292998527162143030538756683302),
            complex(-0.56599287320652734292869884280802459698927645e-12,
                0.0566481651760675042929985271621430305387566833029),
            complex(0.0796884251721652215687859778119964009569455462,
                1.11474461817561675017794941973556302717225126e-22),
            complex(0.07817195821247357458545539935996687005781943386550,
                -0.01093913670103576690766705513142246633056714279654),
            complex(0.04670032980990449912809326141164730850466208439937,
                0.03944038961933534137558064191650437353429669886545),
            complex(0.36787944117144232159552377016146086744581113103176,
                0.60715770584139372911503823580074492116122092866515),
            complex(0,
                0.010259688805536830986089913987516716056946786526145),
            complex(0.99004983374916805357390597718003655777207908125383,
                -0.11208866436449538036721343053869621153527769495574),
            complex(0.99999999999999999999999999999999999999990000,
                1.12837916709551257389615890312154517168802603e-20),
            complex(0.999999999999943581041645226871305192054749891144158,
                0),
            complex(0.0110604154853277201542582159216317923453996211744250,
                0)]#,
#            complex(0,0),
#            complex(0,0),
#            complex(0,0),
#            complex(inf,0),
#            complex(0,0),
#            complex(nan,nan),
#            complex(nan,nan),
#            complex(nan,nan),
#            complex(nan,0),
#            complex(nan,nan),
#            complex(nan,nan)]
        errmax = 0
        i = 0
        nfail = 0
        while i < len(ztst):
#        for (int i = 0; i < NTST; ++i) {
            fw = faddeeva(ztst[i])
            re_err = relerr(wans[i].real, fw.real)
            im_err = relerr(wans[i].imag, fw.imag)
            if verbose:
                print("w(%g%+gi) = %g%+gi (vs. %g%+gi), re/im rel. err. = %0.2g/%0.2g)" % (
                             ztst[i].real,ztst[i].imag, fw.real, fw.imag, wans[i].real, wans[i].imag,
                             re_err, im_err))
            if (re_err > errmax):
                errmax = re_err
            if (im_err > errmax):
                errmax = im_err
            if re_err > 1e-13 or im_err > 1e-13:
                nfail += 1
            i += 1
        if verbose:
            if (errmax > 1e-13):
                print("FAILURE -- relative error {0:e}} too large!\n".format(errmax))
            else:
                print("SUCCESS (max relative error = {0:e})\n".format(errmax))
            print("NFAIL =", nfail)

def test_faddeeva_real(verbose=False):
        ztst = [
            complex(624.2,-0.26123),
            complex(-0.4,3.),
            complex(0.6,2.),
            complex(-1.,1.),
            complex(-1.,-9.),
            complex(-1.,9.),
            complex(-0.0000000234545,1.1234),
            complex(-3.,5.1),
            complex(-53,30.1),
            complex(0.0,0.12345),
            complex(11,1),
            complex(-22,-2),
            complex(9,-28),
            complex(21,-33),
            complex(1e5,1e5),
            complex(1e14,1e14),
            complex(-3001,-1000),
            complex(1e160,-1e159),
            complex(-6.01,0.01),
            complex(-0.7,-0.7),
            complex(2.611780000000000e+01, 4.540909610972489e+03),
            complex(0.8e7,0.3e7),
            complex(-20,-19.8081),
            complex(1e-16,-1.1e-16),
            complex(2.3e-8,1.3e-8),
            complex(6.3,-1e-13),
            complex(6.3,1e-20),
            complex(1e-20,6.3),
            complex(1e-20,16.3),
            complex(9,1e-300),
            complex(6.01,0.11),
            complex(8.01,1.01e-10),
            complex(28.01,1e-300),
            complex(10.01,1e-200),
            complex(10.01,-1e-200),
            complex(10.01,0.99e-10),
            complex(10.01,-0.99e-10),
            complex(1e-20,7.01),
            complex(-1,7.01),
            complex(5.99,7.01),
            complex(1,0),
            complex(55,0),
            complex(-0.1,0),
            complex(1e-20,0),
            complex(0,5e-14),
            complex(0,51)]#,
#            complex(inf,0),
#            complex(-inf,0),
#            complex(0,inf),
#            complex(0,-inf),
#            complex(inf,inf),
#            complex(inf,-inf),
#            complex(nan,nan),
#            complex(nan,0),
#            complex(0,nan),
#            complex(nan,inf),
#            complex(inf,nan)]
        wans = [
            complex(-3.78270245518980507452677445620103199303131110e-7,
                0.000903861276433172057331093754199933411710053155),
            complex(0.1764906227004816847297495349730234591778719532788,
                -0.02146550539468457616788719893991501311573031095617),
            complex(0.2410250715772692146133539023007113781272362309451,
                0.06087579663428089745895459735240964093522265589350),
            complex(0.30474420525691259245713884106959496013413834051768,
                -0.20821893820283162728743734725471561394145872072738),
            complex(7.317131068972378096865595229600561710140617977e34,
                8.321873499714402777186848353320412813066170427e34),
            complex(0.0615698507236323685519612934241429530190806818395,
                -0.00676005783716575013073036218018565206070072304635),
            complex(0.3960793007699874918961319170187598400134746631,
                -5.593152259116644920546186222529802777409274656e-9),
            complex(0.08217199226739447943295069917990417630675021771804,
                -0.04701291087643609891018366143118110965272615832184),
            complex(0.00457246000350281640952328010227885008541748668738,
                -0.00804900791411691821818731763401840373998654987934),
            complex(0.8746342859608052666092782112565360755791467973338452,
                0.),
            complex(0.00468190164965444174367477874864366058339647648741,
                0.0510735563901306197993676329845149741675029197050),
            complex(-0.0023193175200187620902125853834909543869428763219,
                -0.025460054739731556004902057663500272721780776336),
            complex(9.11463368405637174660562096516414499772662584e304,
                3.97101807145263333769664875189354358563218932e305),
            complex(-4.4927207857715598976165541011143706155432296e281,
                -2.8019591213423077494444700357168707775769028e281),
            complex(2.820947917809305132678577516325951485807107151e-6,
                2.820947917668257736791638444590253942253354058e-6),
            complex(2.82094791773878143474039725787438662716372268e-15,
                2.82094791773878143474039725773333923127678361e-15),
            complex(-0.0000563851289696244350147899376081488003110150498,
                -0.000169211755126812174631861529808288295454992688),
            complex(-5.586035480670854326218608431294778077663867e-162,
                5.586035480670854326218608431294778077663867e-161),
            complex(0.00016318325137140451888255634399123461580248456,
                -0.095232456573009287370728788146686162555021209999),
            complex(0.69504753678406939989115375989939096800793577783885,
                -1.8916411171103639136680830887017670616339912024317),
            complex(0.0001242418269653279656612334210746733213167234822,
                7.145975826320186888508563111992099992116786763e-7),
            complex(2.318587329648353318615800865959225429377529825e-8,
                6.182899545728857485721417893323317843200933380e-8),
            complex(-0.0133426877243506022053521927604277115767311800303,
                -0.0148087097143220769493341484176979826888871576145),
            complex(1.00000000000000012412170838050638522857747934,
                1.12837916709551279389615890312156495593616433e-16),
            complex(0.9999999853310704677583504063775310832036830015,
                2.595272024519678881897196435157270184030360773e-8),
            complex(-1.4731421795638279504242963027196663601154624e-15,
                0.090727659684127365236479098488823462473074709),
            complex(5.79246077884410284575834156425396800754409308e-18,
                0.0907276596841273652364790985059772809093822374),
            complex(0.0884658993528521953466533278764830881245144368,
                1.37088352495749125283269718778582613192166760e-22),
            complex(0.0345480845419190424370085249304184266813447878,
                2.11161102895179044968099038990446187626075258e-23),
            complex(6.63967719958073440070225527042829242391918213e-36,
                0.0630820900592582863713653132559743161572639353),
            complex(0.00179435233208702644891092397579091030658500743634,
                0.0951983814805270647939647438459699953990788064762),
            complex(9.09760377102097999924241322094863528771095448e-13,
                0.0709979210725138550986782242355007611074966717),
            complex(7.2049510279742166460047102593255688682910274423e-304,
                0.0201552956479526953866611812593266285000876784321),
            complex(3.04543604652250734193622967873276113872279682e-44,
                0.0566481651760675042930042117726713294607499165),
            complex(3.04543604652250734193622967873276113872279682e-44,
                0.0566481651760675042930042117726713294607499165),
            complex(0.5659928732065273429286988428080855057102069081e-12,
                0.056648165176067504292998527162143030538756683302),
            complex(-0.56599287320652734292869884280802459698927645e-12,
                0.0566481651760675042929985271621430305387566833029),
            complex(0.0796884251721652215687859778119964009569455462,
                1.11474461817561675017794941973556302717225126e-22),
            complex(0.07817195821247357458545539935996687005781943386550,
                -0.01093913670103576690766705513142246633056714279654),
            complex(0.04670032980990449912809326141164730850466208439937,
                0.03944038961933534137558064191650437353429669886545),
            complex(0.36787944117144232159552377016146086744581113103176,
                0.60715770584139372911503823580074492116122092866515),
            complex(0,
                0.010259688805536830986089913987516716056946786526145),
            complex(0.99004983374916805357390597718003655777207908125383,
                -0.11208866436449538036721343053869621153527769495574),
            complex(0.99999999999999999999999999999999999999990000,
                1.12837916709551257389615890312154517168802603e-20),
            complex(0.999999999999943581041645226871305192054749891144158,
                0),
            complex(0.0110604154853277201542582159216317923453996211744250,
                0)]#,
#            complex(0,0),
#            complex(0,0),
#            complex(0,0),
#            complex(inf,0),
#            complex(0,0),
#            complex(nan,nan),
#            complex(nan,nan),
#            complex(nan,nan),
#            complex(nan,0),
#            complex(nan,nan),
#            complex(nan,nan)]
        errmax = 0
        i = 0
        nfail = 0
        while i < len(ztst):
#        for (int i = 0; i < NTST; ++i) {
            fw = faddeeva_real(ztst[i])
            re_err = relerr(wans[i].real, fw)
            if verbose:
                print("w(%g%+gi) = %g (vs. %g), re/im rel. err. = %0.2g)" % (
                         ztst[i].real,ztst[i].imag, fw, wans[i].real, re_err))
            if (re_err > errmax):
                errmax = re_err
            if re_err > 1e-13:
                nfail += 1
            i += 1
        if verbose:
            if (errmax > 1e-13):
                print("FAILURE -- relative error {0:e}} too large!\n".format(errmax))
            else:
                print("SUCCESS (max relative error = {0:e})\n".format(errmax))
            print("NFAIL =", nfail)

test_faddeeva(verbose=True)
test_faddeeva_real(verbose=True)

print(timeit.timeit('test_faddeeva()', globals=globals(), number=1000))
print(timeit.timeit('test_faddeeva_real()', globals=globals(), number=1000))
