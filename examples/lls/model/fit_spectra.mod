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
  ../data/HI_LLS_17p0.dat   specid=0   fitrange=[905.0,920.0]   resolution=vfwhm(7.0va)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  fix voigt temperature True
  emission
    legendre 1.0   0.01   0.01    scale=[1.0,1.0,1.0]   specid=0
  absorption
    voigt   ion=1H_I   17.0cd    0.0rd    10.0   30000.0   specid=0
    phionxs   ion=1H_I   17.0cd    0.0rd   specid=0
model end
