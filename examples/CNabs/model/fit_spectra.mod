run blind False
chisq ftol 1.0E-10
chisq atol 0.001
chisq miniter 10
chisq maxiter 1000
out fits True
plot dims 2x2
plot fits True
plot labels True
#sim perturb 100

data read
  ../data/CN_test.dat   specid=0   fitrange=[3874.8,3876.2]   resolution=vfwhm(0.9VA)   columns=[wave,flux,error]	plotone=True   label=CN
data end

model read
  fix voigt temperature True
  lim voigt bturb [0.01,None]
  emission
    legendre 1.0   0.0   0.0  0.0   0.0    scale=[1.0,1.0,1.0,1.0,1.0]   specid=0
  absorption
# COMPONENT 1
# R(1)
    voigt   ion=1214CN_BX00N1-2J0.5   11.3n1a    0.0za    0.4da   0.0TFIX   specid=0
    voigt   ion=1214CN_BX00N1-2J1.5   11.5n1b    0.0za    0.4da   0.0TFIX   specid=0
# R(0)
    voigt   ion=1214CN_BX00N0-1J0.5   11.5n0    0.0za    0.4da   0.0TFIX   specid=0
    variable  2.7t01  specid=0
model end

link read
# LTE assumption for R(1) lower levels
  n1b(n1a) = numpy.log10(2 * 10.0**n1a)
  n0(n1a,t01) = numpy.log10(3*(10.0**n1a)/(numpy.exp(-5.4312/t01)*(1 + 2*numpy.exp(-0.0157/t01))))
link end
