run blind False
run datadirc ../data/
chisq ftol 1.0E-10
chisq atol 0.001
chisq miniter 10
chisq maxiter 1000
out fits True
out overwrite True
plot dims 2x2
plot fits False
plot labels True

data read
  OI_SiII_thermal.dat   specid=0   fitrange=[1301.0,1305.0]   resolution=vfwhm(3.0VA)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  lim voigt bturb [0.001,None]
  emission
    legendre 1.0   0.01   0.01    scale=[1.0,1.0,1.0]   specid=0
  absorption
    voigt   ion=16O_I   14.0    0.0    0.5da   20000.0ta   specid=0
    voigt   ion=28Si_II 13.0    0.0    0.5da   20000.0ta   specid=0
model end
