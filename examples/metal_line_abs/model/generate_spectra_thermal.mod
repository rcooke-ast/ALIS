generate data True
generate overwrite True
generate pixelsize 0.5
generate peaksnr 200
generate skyfrac 0.001
run blind False
out fits True
plot dims 2x2
plot fits True
plot labels True

data read
  ../data/OI_SiII_thermal.dat   specid=0   fitrange=[1300.0,1306.0]   resolution=vfwhm(3.0VA)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  lim voigt bturb [None,None]
  emission
    constant 1.0CONST
  absorption
    voigt   ion=16O_I   14.0    0.0ra    0.0da   10000.0ta   specid=0
    voigt   ion=28Si_II/16O_I   -1.0    specid=0
model end
