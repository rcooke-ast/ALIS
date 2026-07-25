run blind False
run datadirc ../data/
chisq ftol 1.0E-10
chisq atol 0.001
chisq miniter 10
chisq maxiter 1000
out fits True
out overwrite True
out plots fit_spectra.pdf
plot dims 2x2
plot fits False
plot labels True

data read
  Ha.dat   specid=0   loadrange=all  fitrange=[6545.0,6575.0]   resolution=vfwhm(7.0VA)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  fix voigt temperature True
  emission
    lineemission  ion=1H_I_6563.0  14.0    0.0ra    5.0da   8000.0TA  1000000000000.0   specid=0
model end
