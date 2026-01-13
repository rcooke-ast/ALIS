generate data True
generate overwrite True
generate pixelsize 50.0
generate peaksnr 100
generate skyfrac 0.01
run blind False
out fits True
plot dims 2x2
plot fits True
plot labels True

data read
  ../data/HaNII.dat   specid=0   fitrange=[6550.0,6600.0]   resolution=vfwhm(150.0VA)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  emission
    gaussian  10.0   0.0   30.0  specid=0  wave=6563.0
model end
