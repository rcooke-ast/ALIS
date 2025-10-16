import numpy as np
from alis import almsgs
from alis import alfunc_base
msgs=almsgs.msgs()


class Gaussian(alfunc_base.Base) :
    """
    Returns a 1-dimensional gaussian of form:
    p[0] = amplitude
    p[1] = x offset
    p[2] = dispersion (sigma)
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'gaussian'									# ID string for this class
        self._pnumr   = 3											# Total number of parameters fed in
        self._keywd   = dict({'specid':[], 'continuum':False, 'blind':False, 'wave':-1.0, 'IntFlux':False, 'blindseed':0,  'blindrange':[]})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'specid':0,  'continuum':0,     'blind':0,     'wave':1,    'IntFlux':0, 'blindseed':0,  'blindrange':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'specid':"", 'continuum':"",    'blind':"",    'wave':"",   'IntFlux':"", 'blindseed':"",  'blindrange':""})			# Format for the keyword. "" is the Default setting
        self._parid   = ['amplitude', 'redshift', 'dispersion']		# Name of each parameter
        self._defpar  = [ 0.0,         0.0,        100.0 ]			# Default values for parameters that are not provided
        self._fixpar  = [ None,       None,      None ]				# By default, should these parameters be fixed?
        self._limited = [ [1  ,0  ],   [0  ,0  ], [1      ,0  ] ]	# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0],   [0.0,0.0], [0.01,0.0] ]	# What should these limiting values be
        self._svfmt   = [ "{0:.8g}",  "{0:.8g}", "{0:.8g}"]			# Specify the format used to print or save output
        self._prekw   = []											# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        if getinst: return

    def call_CPU(self, x, p, ae='em', mkey=None, ncpus=1):
        """
        Define the functional form of the model
        --------------------------------------------------------
        x  : array of wavelengths
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        def model(par, karr):
            """
            Define the model here
            """
            if karr['IntFlux']:
                return (par[0]/(np.sqrt(2.0*np.pi)*par[2]))*np.exp(-(x-par[1])**2/(2.0*(par[2]**2)))
            else:
                return par[0]*np.exp(-(x-par[1])**2/(2.0*(par[2]**2)))
        #############
        yout = np.zeros((p.shape[0],x.size))
        for i in range(p.shape[0]):
            if ae == 'zl':
                yout[i, :] = model(p[i, :], karr=mkey)
            else:
                yout[i, :] = model(p[i, :], karr=mkey[i])
        if ae == 'em': return yout.sum(axis=0)
        else: return yout.prod(axis=0)

    def parin(self, i, par, parb):
        """
        This routine converts a parameter in the input model file
        to the parameter used in 'call'
        --------------------------------------------------------
        When writing a new function, one should change how each
        input parameter 'par' is converted into a parameter used
        in the function specified by 'call'
        --------------------------------------------------------
        """
        if   i == 0: pin = par
        elif i == 1:
            pin = parb['ap_1a'] * (1.0 + par)
        elif i == 2: pin = parb['ap_2a'] * par/299792.458
#		elif i == 2: pin = par
        return pin

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, ddpid=None, getinfl=False):
        """
        Return the parameters for a Gaussian function to be used by 'call'
        The only thing that should be changed here is the parb values
        and possibly the nexbin details...
        """
        levadd=0
        params=np.zeros(self._pnumr)
        parinf=[]
        for i in range(self._pnumr):
            lnkprm = None
            parb = dict({'ap_1a':self._keywd['wave'], 'ap_2a':params[1]})
            if mp['mtie'][ival][i] >= 0:
                getid = mp['tpar'][mp['mtie'][ival][i]][1]
            elif mp['mtie'][ival][i] <= -2:
                if len(mp['mlnk']) == 0:
                    lnkprm = mp['mpar'][ival][i]
                else:
                    for j in range(len(mp['mlnk'])):
                        if mp['mlnk'][j][0] == mp['mtie'][ival][i]:
                            cmd = 'lnkprm = ' + mp['mlnk'][j][1]
                            namespace = dict({'p': p})
                            exec(cmd, namespace)
                            lnkprm = namespace['lnkprm']
                levadd += 1
            else:
                getid = level+levadd
                levadd+=1
            if lnkprm is None:
                params[i] = self.parin(i, p[getid], parb)
                if mp['mfix'][ival][i] == 0: parinf.append(getid)
            else:
                params[i] = lnkprm
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if params[2] == 0.0: msgs.error("Cannot calculate "+self._idstr+" subpixellation -- width = 0.0")
            if nexbin[0] == "km/s": return params, int(round(parb['ap_2a']*nexbin[1]/(299792.458*params[2]) + 0.5))
            elif nexbin[0] == "A" : return params, int(round(nexbin[1]/params[2] + 0.5))
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getinfl: return params, parinf
        else: return params


class GaussianConstant(alfunc_base.Base) :
    """
    Returns a 1-dimensional gaussian of form:
    p[0] = constant offset
    p[1] = amplitude
    p[2] = x offset
    p[3] = dispersion (sigma)
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'gaussianconst'									# ID string for this class
        self._pnumr   = 4											# Total number of parameters fed in
        self._keywd   = dict({'specid':[], 'continuum':False, 'blind':False, 'wave':-1.0, 'IntFlux':False})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'specid':0,  'continuum':0,     'blind':0,     'wave':1,    'IntFlux':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'specid':"", 'continuum':"",    'blind':"",    'wave':"",   'IntFlux':""})			# Format for the keyword. "" is the Default setting
        self._parid   = ['offset', 'amplitude', 'redshift', 'dispersion']		# Name of each parameter
        self._defpar  = [ 0.0,      0.0,         0.0,        100.0 ]			# Default values for parameters that are not provided
        self._fixpar  = [ None,     None,       None,      None ]				# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ],   [1  ,0  ],   [0  ,0  ], [1      ,0  ] ]	# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0],   [0.0,0.0],   [0.0,0.0], [1.0,0.0] ]	# What should these limiting values be
        self._svfmt   = [ "{0:.8g}",  "{0:.8g}",  "{0:.8g}", "{0:.8g}"]			# Specify the format used to print or save output
        self._prekw   = []											# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        if getinst: return

    def call_CPU(self, x, p, ae='em', mkey=None, ncpus=1):
        """
        Define the functional form of the model
        --------------------------------------------------------
        x  : array of wavelengths
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        def model(par, karr):
            """
            Define the model here
            """
            if karr['IntFlux']:
                return par[0] + (par[1]/(np.sqrt(2.0*np.pi)*par[3]))*np.exp(-(x-par[2])**2/(2.0*(par[3]**2)))
            else:
                return par[0] + par[1]*np.exp(-(x-par[2])**2/(2.0*(par[3]**2)))
        #############
        yout = np.zeros((p.shape[0],x.size))
        for i in range(p.shape[0]):
            if ae=='zl': yout[i,:] = model(p[i,:], karr=mkey)
            else: yout[i,:] = model(p[i,:], karr=mkey[i])
        if ae == 'em': return yout.sum(axis=0)
        else: return yout.prod(axis=0)

    def parin(self, i, par, parb):
        """
        This routine converts a parameter in the input model file
        to the parameter used in 'call'
        --------------------------------------------------------
        When writing a new function, one should change how each
        input parameter 'par' is converted into a parameter used
        in the function specified by 'call'
        --------------------------------------------------------
        """
        if   i == 0: pin = par
        elif i == 1: pin = par
        elif i == 2: pin = parb['ap_1a'] * (1.0 + par)
        elif i == 3: pin = parb['ap_2a'] * par/299792.458
        return pin

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, ddpid=None, getinfl=False):
        """
        Return the parameters for a Gaussian function to be used by 'call'
        The only thing that should be changed here is the parb values
        and possibly the nexbin details...
        """
        levadd=0
        params=np.zeros(self._pnumr)
        parinf=[]
        for i in range(self._pnumr):
            lnkprm = None
            parb = dict({'ap_1a':self._keywd['wave'], 'ap_2a':params[2]})
            if mp['mtie'][ival][i] >= 0:
                getid = mp['tpar'][mp['mtie'][ival][i]][1]
            elif mp['mtie'][ival][i] <= -2:
                if len(mp['mlnk']) == 0:
                    lnkprm = mp['mpar'][ival][i]
                else:
                    for j in range(len(mp['mlnk'])):
                        if mp['mlnk'][j][0] == mp['mtie'][ival][i]:
                            cmd = 'lnkprm = ' + mp['mlnk'][j][1]
                            namespace = dict({'p': p})
                            exec(cmd, namespace)
                            lnkprm = namespace['lnkprm']
                levadd += 1
            else:
                getid = level+levadd
                levadd+=1
            if lnkprm is None:
                params[i] = self.parin(i, p[getid], parb)
                if mp['mfix'][ival][i] == 0: parinf.append(getid)
            else:
                params[i] = lnkprm
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if params[2] == 0.0: msgs.error("Cannot calculate "+self._idstr+" subpixellation -- width = 0.0")
            if nexbin[0] == "km/s": return params, int(round(parb['ap_2a']*nexbin[1]/(299792.458*params[2]) + 0.5))
            elif nexbin[0] == "A" : return params, int(round(nexbin[1]/params[2] + 0.5))
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getinfl: return params, parinf
        else: return params



class GaussianConstantHelium(alfunc_base.Base) :
    """
    Returns a 1-dimensional gaussian of form:
    p[0] = constant offset
    p[1] = amplitude
    p[2] = redshift
    p[3] = dispersion (sigma)
    p[4] = relative strength of weak 10833 line relative to the sum of the two strong lines at 10833
    p[5] = relative strength of the 3889 line relative to the sum of the two strong lines at 10833 line
    p[6] = relative strength of the 3188 line relative to the sum of the two strong lines at 10833 line
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'gaussianconsthelium'									# ID string for this class
        self._pnumr   = 7											# Total number of parameters fed in
        self._keywd   = dict({'specid':[], 'continuum':False, 'blind':False, 'wave':-1.0, 'IntFlux':False})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'specid':0,  'continuum':0,     'blind':0,     'wave':0,    'IntFlux':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'specid':"", 'continuum':"",    'blind':"",    'wave':"",   'IntFlux':""})			# Format for the keyword. "" is the Default setting
        self._parid   = ['offset', 'amplitude', 'redshift', 'dispersion', 'amplitude2', 'amplitude3889', 'amplitude3188']		# Name of each parameter
        self._defpar  = [ 0.0,      0.0,         0.0,        100.0,        0.0,        0.0,        0.0 ]			# Default values for parameters that are not provided
        self._fixpar  = [ None,     None,       None,      None,          None,      None,          None ]				# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ],   [1  ,0  ],   [0  ,0  ], [1      ,0  ], [1  ,1  ], [1  ,1  ], [1  ,1  ] ]	# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0],   [0.0,0.0],   [0.0,0.0], [1.0,0.0],   [0.0,0.125],   [0.0,0.125],   [0.0,0.125] ]	# What should these limiting values be
        self._svfmt   = [ "{0:.8g}",  "{0:.8g}",  "{0:.8g}", "{0:.8g}", "{0:.8g}", "{0:.8g}", "{0:.8g}"]			# Specify the format used to print or save output
        self._prekw   = []											# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        if getinst: return

    def call_CPU(self, x, p, ae='em', mkey=None, ncpus=1):
        """
        Define the functional form of the model
        --------------------------------------------------------
        x  : array of wavelengths
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        def model(par, karr):
            """
            Define the model here
            """
            if karr['IntFlux']:
                print("Not ready for integrated flux yet!")
                assert False
                return par[0] + (par[1]/(np.sqrt(2.0*np.pi)*par[3]))*np.exp(-(x-par[2])**2/(2.0*(par[3]**2)))
            else:
                #modv = par[0] + par[1]*np.exp(-(x-par[2])**2/(2.0*(par[3]**2))) + par[1]*par[4]*np.exp(-(x-par[2]+1.2153364838268317)**2/(2.0*(par[3]**2)))
                # fval weighted wavelengths: 3188.665406014746, 3889.7448067333644, 10833.272806483827 (this is the two strongest lines)
                lcen10833  = 10833.272806483827*(1.0 + par[2])
                lcen10833b = 10832.05747*(1.0 + par[2])
                lcen3889   = 3889.7448067333644*(1.0 + par[2])
                lcen3188   = 3188.665406014746*(1.0 + par[2])
                wid10833   = lcen10833 * par[3] / 299792.458
                wid10833b  = lcen10833b * par[3] / 299792.458
                wid3889    = lcen3889 * par[3] / 299792.458
                wid3188    = lcen3188 * par[3] / 299792.458
                modv = par[0]
                modv += par[1]*np.exp(-(x-lcen10833)**2/(2.0*(wid10833**2)))
                modv += par[1]*par[4]*np.exp(-(x-lcen10833b)**2/(2.0*(wid10833b**2)))
                modv += par[1]*par[5]*np.exp(-(x-lcen3889)**2/(2.0*(wid3889**2)))
                modv += par[1]*par[6]*np.exp(-(x-lcen3188)**2/(2.0*(wid3188**2)))
                return modv

        #############
        yout = np.zeros((p.shape[0],x.size))
        for i in range(p.shape[0]):
            if ae=='zl': yout[i,:] = model(p[i,:], karr=mkey)
            else: yout[i,:] = model(p[i,:], karr=mkey[i])
        if ae == 'em': return yout.sum(axis=0)
        else: return yout.prod(axis=0)

    def parin(self, i, par, parb):
        """
        This routine converts a parameter in the input model file
        to the parameter used in 'call'
        --------------------------------------------------------
        When writing a new function, one should change how each
        input parameter 'par' is converted into a parameter used
        in the function specified by 'call'
        --------------------------------------------------------
        """
        if   i == 0: pin = par
        elif i == 1: pin = par
        elif i == 2: pin = par#parb['ap_1a'] * (1.0 + par)
        elif i == 3: pin = par
        elif i == 4: pin = par
        elif i == 5: pin = par
        elif i == 6: pin = par
        return pin

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, ddpid=None, getinfl=False):
        """
        Return the parameters for a Gaussian function to be used by 'call'
        The only thing that should be changed here is the parb values
        and possibly the nexbin details...
        """
        levadd=0
        params=np.zeros(self._pnumr)
        parinf=[]
        for i in range(self._pnumr):
            lnkprm = None
            # fval weighted wavelengths: 3188.665406014746, 3889.7448067333644, 10833.272806483827
            # parb = dict({'ap_1a':self._keywd['wave'], 'ap_2a':params[2]})
            parb = dict({'ap_1a':10833.272806483827, 'ap_2a':10833.272806483827*(1+params[2])})
            if mp['mtie'][ival][i] >= 0:
                getid = mp['tpar'][mp['mtie'][ival][i]][1]
            elif mp['mtie'][ival][i] <= -2:
                if len(mp['mlnk']) == 0:
                    lnkprm = mp['mpar'][ival][i]
                else:
                    for j in range(len(mp['mlnk'])):
                        if mp['mlnk'][j][0] == mp['mtie'][ival][i]:
                            cmd = 'lnkprm = ' + mp['mlnk'][j][1]
                            namespace = dict({'p': p})
                            exec(cmd, namespace)
                            lnkprm = namespace['lnkprm']
                levadd += 1
            else:
                getid = level+levadd
                levadd+=1
            if lnkprm is None:
                params[i] = self.parin(i, p[getid], parb)
                if mp['mfix'][ival][i] == 0: parinf.append(getid)
            else:
                params[i] = lnkprm
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if params[3] == 0.0: msgs.error("Cannot calculate "+self._idstr+" subpixellation -- width = 0.0")
            if nexbin[0] == "km/s": return params, int(round(parb['ap_2a']*nexbin[1]/(299792.458*params[3]) + 0.5))
            elif nexbin[0] == "A" : return params, int(round(nexbin[1]/params[3] + 0.5))
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getinfl: return params, parinf
        else: return params
