run blind False
out fits True
plot dims 2x2
plot fits True
plot labels True

data read
  ../data/OI_SiII.dat   specid=0   fitrange=[1301.0,1305.0]  loadrange=all  resolution=lsfspline(4.0va,0.0LFIX,0.0vva,0.0vvb,0.5vvc,1.0CFIX,0.5vvd,0.0vve,0.0vvf,0.0RFIX;locations:-10.0,-5.0,-3.0,-1.0,0.0,1.0,3.0,5.0,10.0)   columns=[wave,flux,error]	plotone=True   label=OI_SiII
data end

model read
  emission
    constant 1.0CONST  specid=0
  absorption
    voigt   ion=16O_I   14.0    0.0ra    5.0da   8000.0ta   specid=0
    voigt   ion=28Si_II/16O_I   -1.0    specid=0
model end
