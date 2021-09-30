run blind False
generate data True
generate pixelsize 2.5
generate peaksnr 150.0
generate skyfrac 0.001
plot fits True
out fits True
out overwrite True

data read
  ../data/summed_coldens_H_I_1215.dat  specid=0  fitrange=[1213.0,1217.0]  resolution=vfwhm(7.0va)  columns=[wave:0,flux:1,error:2]  plotone=True  label=H_I_1215
data end

model read
  fix vfwhm value True
  emission
    constant  1.0CNS  specid=0
  absorption
    voigt ion=1H_I  12.0na   0.00000   3.5   0.000TEMP   specid=0
    voigt ion=1H_I  11.9nb   0.00010   5.0   0.000TEMP   specid=0
    voigt ion=1H_I  11.8nc   0.00020   2.4   0.000TEMP   specid=0
    voigt ion=1H_I  12.3nd   0.00007   8.2   0.000TEMP   specid=0
    voigt ion=1H_I  11.7ne   0.00032   4.5   0.000TEMP   specid=0
    variable  12.7nf specid=0
model end

link read
  na(nb,nc,nd,ne,nf) = numpy.log10(10.0**nf - (10.0**nb + 10.0**nc + 10.0**nd + 10.0**ne))
link end