generate data True
generate overwrite True
generate pixelsize 2.5
generate peaksnr 20
generate skyfrac 0.02
run blind False
out fits True
plot dims 2x2
plot fits True
plot labels True

data read
  ../data/Ha.dat   specid=0   fitrange=[6540.0,6580.0]   resolution=vfwhm(7.0VA)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  emission
    lineemission  ion=1H_I_6563.0  14.0    0.0ra    5.0da   8000.0ta  1000000000000.0   specid=0
model end
