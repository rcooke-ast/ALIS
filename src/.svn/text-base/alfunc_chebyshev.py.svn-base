import numpy as np
import almsgs
import alfunc_polynomial
msgs=almsgs.msgs()

class Chebyshev(alfunc_polynomial.Polynomial) :
	"""
	Returns a Chebyshev polynomial of the first kind:
	p[0] = coefficient of the term :  1
	p[1] = coefficient of the term :  x
	p[2] = coefficient of the term :  2*x**2 - 1.0
	...

	Note that I've opted to create just a single polynomial function where an
	arbitrary number of coefficients can by specified. The `downside' is that
	you cannot specify the parid, fixpar, limited, or limits parameters. Although
	these commands will work, the limits you place will be applied to all
	coefficients, which may not be so bad, if you use the scale keyword to
	scale your coefficients to be of the same magnitude. If you want the ability
	to limit your polynomial coefficients, another way around this would be to
	write your own polynomial function for the polynomial order that interests
	you.
	"""
	def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
		self._idstr   = 'chebyshev'								# ID string for this class
		self._pnumr   = 1											# Total number of parameters fed in
		self._keywd   = dict({'specid':[], 'blind':False, 'scale':[]})			# Additional arguments to describe the model --- 'input' cannot be used as a keyword
		self._keych   = dict({'specid':0,  'blind':0,     'scale':0})			# Require keywd to be changed (1 for yes, 0 for no)
		self._keyfm   = dict({'specid':"", 'blind':"",    'scale':""})		# Format for the keyword. "" is the Default setting
		self._parid   = ['coefficient']				# Name of each parameter
		self._defpar  = [ 0.0 ]						# Default values for parameters that are not provided
		self._fixpar  = [ None ]					# By default, should these parameters be fixed?
		self._limited = [ [0  ,0  ] ]				# Should any of these parameters be limited from below or above
		self._limits  = [ [0.0,0.0] ]				# What should these limiting values be
		self._svfmt   = [ "{0:.8g}" ]				# Specify the format used to print or save output
		self._prekw   = []							# Specify the keywords to print out before the parameters
		# DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
		tempinput = self._parid+self._keych.keys()                             #
		self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
		########################################################################
		self._verbose = verbose
		# Set the atomic data
		self._atomic = atomic
		if getinst: return

	def call_CPU(self, x, p, ae='em', mkey=None, ncpus=1):
		"""
		Define the functional form of the model for the CPU
		--------------------------------------------------------
		x  : array of wavelengths
		p  : array of parameters for this model
		--------------------------------------------------------
		"""
		def model(par):
			"""
			Define the model here
			"""
			modret = np.ones(x.size)*par[0]
			xt = 2.0*x/(np.max(x)-np.min(x))
			xt = xt - np.min(xt) - 1.0
			for m in range(1,len(par)):
				if   m == 1 : modret += par[m]*(xt)
				elif m == 2 : modret += par[m]*(  2.0*xt**2 -   1.0)
				elif m == 3 : modret += par[m]*(  4.0*xt**3 -   3.0*xt)
				elif m == 4 : modret += par[m]*(  8.0*xt**4 -   8.0*xt**2 +   1.0)
				elif m == 5 : modret += par[m]*( 16.0*xt**5 -  20.0*xt**3 +   5.0*xt)
				elif m == 6 : modret += par[m]*( 32.0*xt**6 -  48.0*xt**4 +  18.0*xt**2 -   1.0)
				elif m == 7 : modret += par[m]*( 64.0*xt**7 - 112.0*xt**5 +  56.0*xt**3 -   7.0*xt)
				elif m == 8 : modret += par[m]*(128.0*xt**8 - 256.0*xt**6 + 160.0*xt**4 -  32.0*xt**2 + 1.0)
				elif m == 9 : modret += par[m]*(256.0*xt**9 - 576.0*xt**7 + 432.0*xt**5 - 120.0*xt**3 + 9.0*xt)
				else:
					msgs.bug("Chebyshev polynomials of order 10 and above are not implemented")
					sys.exit()
			return modret
		#############
		yout = np.zeros((p.shape[0],x.size))
		for i in range(p.shape[0]):
			yout[i,:] = model(p[i,:])
		if ae == 'em': return yout.sum(axis=0)
		else: return yout.prod(axis=0)

	def call_GPU(self, x, p, ae='em'):
		"""
		Define the functional form of the model for the GPU
		--------------------------------------------------------
		x  : array of wavelengths
		p  : array of parameters for this model
		--------------------------------------------------------
		"""
		self.call_CPU(x, p, ae=ae)

