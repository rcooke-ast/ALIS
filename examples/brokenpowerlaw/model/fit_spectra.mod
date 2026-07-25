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
  OI_SiII_brokenpowerlaw.dat   specid=0   fitrange=[1301.0,1305.0]   resolution=vfwhm(7.0VA)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  fix voigt temperature True
  emission
    brokenpowerlaw 1.0 0.0 0.0 1303.0LOC 10000.0STR
  absorption
    voigt   ion=16O_I   14.0    0.0ra    1.0da   8000TA   specid=0
    voigt   ion=28Si_II 13.0    0.0ra    1.0da   8000TA   specid=0
model end
