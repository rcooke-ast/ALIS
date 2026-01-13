run blind False
chisq ftol 1.0E-10
chisq atol 0.001
chisq miniter 10
chisq maxiter 1000
out fits True
plot dims 1x1
plot fits True
plot labels True
#sim perturb 100

data read
  ../data/HaNII.dat   specid=0   fitrange=[6550.0,6600.0]   resolution=vfwhm(150.0VA)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  lim gaussian amplitude  None,None
  emission
    gaussian  10.0ha   0.0za   30.0ba  specid=0  wave=6563.0
    gaussian  0.0nii   0.0za   30.0ba  specid=0  wave=6584.0
    variable  0.0ratio
model end

link read
  nii(ratio,ha) = ratio*ha
link end
