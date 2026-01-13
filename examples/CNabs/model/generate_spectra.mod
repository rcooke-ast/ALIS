generate data True
generate overwrite True
generate pixelsize 0.4
generate peaksnr 1000
generate skyfrac 0.001
run blind False
out fits True
plot dims 2x2
plot fits True
plot labels True

data read
  ../data/CN_test.dat   specid=0   fitrange=[3874.5,3876.5]   resolution=vfwhm(1.0VA)   columns=[wave,flux,error]	plotone=True   label=CN
data end

model read
  fix voigt temperature True
  emission
    legendre 1.0   0.0   0.0  0.0   0.0    scale=[1.0,1.0,1.0,1.0,1.0]   specid=0
  absorption
# COMPONENT 1
# R(1)
    voigt   ion=1214CN_BX00N1-2J0.5   11.3n1a    0.0za    0.4da   0.0TFIX   specid=0
    voigt   ion=1214CN_BX00N1-2J1.5   11.5n1b    0.0za    0.4da   0.0TFIX   specid=0
# P(1)
    voigt   ion=1214CN_BX00N1-0J0.5   11.2n1a    0.0za    0.4da   0.0TFIX   specid=0
    voigt   ion=1214CN_BX00N1-0J1.5   12.2n1b    0.0za    0.4da   0.0TFIX   specid=0
# R(0)
    voigt   ion=1214CN_BX00N0-1J0.5   11.5n0    0.0za    0.4da   0.0TFIX   specid=0
    variable  2.7t01  specid=0
model end

link read
# LTE assumption for R(1) lower levels
  n1b(n1a) = numpy.log10(2 * 10.0**n1a)
  n0(n1a,t01) = numpy.log10(3*(10.0**n1a)/(numpy.exp(-5.4312/t01)*(1 + 2*numpy.exp(-0.0157/t01))))
link end
