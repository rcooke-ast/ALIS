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
  ../data/OI_SiII.dat   specid=0   fitrange=[1301.0,1305.0]   resolution=vfwhm(7.0va)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  fix voigt temperature True
  emission
    legendre 1.0   0.01   0.01    scale=[1.0,1.0,1.0]   specid=0
  absorption
  splineabs   ion=16O_I        13.6568339      0.0ZABS  4.048338      0.0000000FIXL       0.0818007      0.2323805      1.0000000      1.0000000FIXC      0.8562307      0.4530773      0.0220766      0.0000000FIXR   specid=0    blind=False      locations=-12.0,-9.0,-6.0,-3.0,0.0,3.0,6.0,9.0,12.0
  splineabs   ion=28Si_II        12.6568339      0.0ZABS  4.048338      0.0000000FIXL       0.0818007      0.2323805      1.0000000      1.0000000FIXC      0.8562307      0.4530773      0.0220766      0.0000000FIXR   specid=0    blind=False      locations=-12.0,-9.0,-6.0,-3.0,0.0,3.0,6.0,9.0,12.0
#    voigt   ion=16O_I   14.0    0.0    0.5da   2.0E4TA   specid=0
#    voigt   ion=28Si_II 13.0    0.0    0.5da   2.0E4TA   specid=0
model end
