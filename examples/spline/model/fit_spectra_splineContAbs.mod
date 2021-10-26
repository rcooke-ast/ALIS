run blind False
chisq ftol 1.0E-10
chisq atol 0.001
chisq miniter 10
chisq maxiter 1000
out fits True
#plot only True
plot dims 2x2
plot fits True
plot labels True
#sim perturb 100

data read
  ../data/OI_SiII.dat   specid=0   fitrange=[1301.0,1305.0]   resolution=vfwhm(7.0VFIX)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  fix voigt temperature True
  emission
    spline 1.0   1.0   1.0  1.0   1.0   specid=0
  absorption
    splineabs   ion=16O_I   16.0    0.0ZABS    5.0   0.0FIXL   0.1   0.5    1.0FIXC   0.5   0.1   0.0FIXR   locations=[-10.0,-5.0,-2.0,0.0,2.0,5.0,10.0]  specid=0
    splineabs   ion=28Si_II/16O_I   -3.0    specid=0
model end
