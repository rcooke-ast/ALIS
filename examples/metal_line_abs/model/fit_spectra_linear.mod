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
#plot only True

data read
  ../data/OI_SiII.dat   specid=0   fitrange=[1301.0,1305.0]   resolution=vfwhm(7.0VA)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  emission
    legendre 1.0   0.01   0.01    scale=[1.0,1.0,1.0]   specid=0
  absorption
    voigt   ion=16O_I   1.0    0.0    5da   8.0E3TA   specid=0  logN=False  ColDensScale=1.0E14
    voigt   ion=28Si_II 13.0    0.0    5da   8.0E3TA   specid=0
model end
