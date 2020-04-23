import os
import sys
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from myobjects import dlas
matplotlib.use('Qt5Agg')
from alis.prepfit import specplot

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	dla = dlas("object1")
	prop = specplot.props(dla)

	# Ignore lines outside of wavelength range
	wmin, wmax = np.min(prop._wave)/(1.0+prop._zabs), np.max(prop._wave)/(1.0+prop._zabs)

	# Load atomic data
	atom = specplot.atomic(wmin=wmin, wmax=wmax)

	spec = Line2D(prop._wave, prop._flux, linewidth=1, linestyle='solid', color='k', drawstyle='steps', animated=True)

	fig, ax = plt.subplots(figsize=(16,9), facecolor="white")
	ax.add_line(spec)
	reg = specplot.SelectRegions(fig.canvas, ax, spec, prop, atom)

	ax.set_title("Press '?' to list the available options")
	#ax.set_xlim((prop._wave.min(), prop._wave.max()))
	ax.set_ylim((0.0, prop._flux.max()))
	plt.show()
