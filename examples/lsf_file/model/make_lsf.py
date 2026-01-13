from matplotlib import pyplot as plt
import numpy as np

numsamp = 4096
AFWHM = 0.05
nsig = 15
Asig = AFWHM / ( 2.0*np.sqrt(2.0*np.log(2.0)) )
diff = nsig*Asig

wav = np.linspace(-diff, +diff, numsamp)
lsf = np.exp(-0.5*(wav/Asig)**2)

plt.plot(wav, lsf, 'k')
plt.show()

fname = "test_lsf.dat"
print(f"Writing file {fname}")
np.savetxt(fname, np.transpose((wav, lsf)))
