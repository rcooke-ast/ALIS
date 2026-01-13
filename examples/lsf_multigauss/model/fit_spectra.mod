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
  ../data/OI_SiII.dat   specid=0   fitrange=[1301.0,1305.0]   resolution=multivfwhm(7.0VA,0.5,7.0,3.0)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  fix voigt temperature True
  emission
    legendre 1.0   0.01   0.01    scale=[1.0,1.0,1.0]   specid=0
  absorption
    voigt   ion=16O_I   14.0    0.0    2.0da   8000.0TA   specid=0
    voigt   ion=28Si_II 13.0    0.0    2.0da   8000.0TA   specid=0
model end
