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

#
# NOTE: There is a bug around the Lyman Limit,
# which is because of the incomplete Lyman series
# used by the code. The phionxs model does not
# replace the model, which might be a solution
# to the bug...
#

data read
  ../data/HI_LLS_17p0.dat   specid=0   fitrange=[900.0,930.0]   resolution=vfwhm(0.0VA)   columns=[wave,flux,error]	plotone=True   label=HI_LymanLimit
data end

model read
  emission
    constant 1.0CONST
  absorption
    voigt   ion=1H_I   17.0cd    0.0rd    10.0   30000.0   specid=0
    phionxs   ion=1H_I   17.0cd    0.0rd   specid=0
model end
