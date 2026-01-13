generate data True
generate overwrite True
generate pixelsize 2.5
generate peaksnr 100
generate skyfrac 0.01
run blind False
out fits True
plot dims 2x2
plot fits True
plot labels True

data read
  ../data/OI_SiII.dat   specid=0   fitrange=[1300.0,1306.0]   resolution=multivfwhm(7.0VA,0.5,7.0,3.0)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  emission
    constant 1.0CONST
  absorption
    voigt   ion=16O_I   14.0    0.0ra    2.0da   8000.0ta   specid=0
    voigt   ion=28Si_II/16O_I   -1.0    specid=0
model end
