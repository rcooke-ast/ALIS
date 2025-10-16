import numpy as np
from alis import almsgs
from alis import alfunc_base
msgs=almsgs.msgs()

class Ashift(alfunc_base.Base) :
    """
    Shifts the model spectrum by p[0] Angstroms:
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'Ashift'			# ID string for this class
        self._pnumr   = 1				# Total number of parameters fed in
        self._keywd   = dict({'blind':False})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'blind':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'blind':""})			# Require keywd to be changed (1 for yes, 0 for no)
        self._parid   = ['value']		# Name of each parameter
        self._defpar  = [ 0.0 ]			# Default values for parameters that are not provided
        self._fixpar  = [ None ]		# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ] ]	# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0] ]	# What should these limiting values be
        self._svfmt   = [ "{0:.7g}" ]	# Specify the format used to print or save output
        self._prekw   = []				# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        if getinst: return

    def call_CPU(self, x, p, ncpus=1, mkey=None):
        """
        Define the functional form of the model
        --------------------------------------------------------
        x  : array of wavelengths
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        return x-p[0]


    def load(self, instr, cntr, mp, specid, forcefix=False):
        """
        Load the parameters in the input model file
        --------------------------------------------------------
        instr: input string for the parameters and keywords of
               model to be loaded (ignoring the identifier at
               the beginning of the model line).
        cntr : The line number of the model (e.g. if it's the
               first model line, cntr=0)
        mp   : modpass --> A dictionary with the details of all
               models read in so far.
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.
        --------------------------------------------------------
        """
        def check_tied_param(ival, cntr, mps, iind):
            havtie = False
            tieval=ival.lstrip('+-.0123456789')
            if tieval[0:2] in ['E+', 'e+', 'E-', 'e-']: # Scientific Notation is used.
                tieval=tieval[2:].lstrip('.0123456789')
            inval=float(ival.rstrip(tieval))
            if len(tieval) == 0: # Parameter is not tied
                mps['mtie'][cntr].append(-1)
                if forcefix:
                    mps['mfix'][cntr].append(1)
                else:
                    mps['mfix'][cntr].append(0)
            else: # parameter is tied
                # Determine if this parameter is fixed
                if tieval[0].isupper() or forcefix: mps['mfix'][cntr].append(1)
                else: mps['mfix'][cntr].append(0)
                # Check if this tieval has been used before
                if len(mps['tpar']) == 0: # If it's the first known tied parameter in the model
                    mps['tpar'].append([])
                    mps['tpar'][0].append(tieval)
                    mps['tpar'][0].append(len(mps['p0']))
                    mps['mtie'][cntr].append(-1) # i.e. not tied to anything
                else:
                    for j in range(0,len(mps['tpar'])):
                        if mps['tpar'][j][0] == tieval:
                            mps['mtie'][cntr].append(j)
                            havtie = True
                    if havtie == False: # create a New tied parameter
                        mps['tpar'].append([])
                        mps['tpar'][-1].append(tieval)
                        mps['tpar'][-1].append(len(mps['p0']))
                        mps['mtie'][cntr].append(-1) # i.e. not tied to anything
            if havtie == False: mps['p0'].append(inval)
            mps['mpar'][cntr].append(inval)
            mps['mlim'][cntr].append([self._limits[iind][i] if self._limited[iind][i]==1 else None for i in range(2)])
            return mps
        ################
        isspl=instr.split()
        # Seperate the parameters from the keywords
        kywrd = []
        keywdk = list(self._keywd.keys())
        keywdk[:] = (kych for kych in keywdk if kych[:] != 'input') # Remove the keyword 'input'
        param = [None for all in range(self._pnumr)]
        parid = [i for i in range(self._pnumr)]
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                kwspl = isspl[i].split('=')
                if kwspl[0] in keywdk:
                    self._keywd['input'][kwspl[0]]=1
                    kywrd.append(isspl[i])
                else: msgs.error("Keyword '"+isspl[i]+"' is unknown for -"+msgs.newline()+self._idstr+"   "+instr)
            else:
                param[i] = isspl[i]
                self._keywd['input'][self._parid[i]]=1
        # Do some quick checks
        if len(param) != self._pnumr:
            msgs.error("Incorrect number of parameters (should be "+str(self._pnumr)+"):"+msgs.newline()+self._idstr+"   "+instr)
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(self._pnumr):
            mp = check_tied_param(param[i], cntr, mp, i)
        # Now load the keywords:
        for i in range(len(kywrd)):
            kwspl = kywrd[i].split('=')
            ksspl = kwspl[1].split(',')
            for j in range(len(ksspl)):
                if type(self._keywd[kwspl[0]]) is int:
                    typeval='integer'
                    self._keywd[kwspl[0]] = int(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is str:
                    typeval='string'
                    self._keywd[kwspl[0]] = kwspl[1]
                elif type(self._keywd[kwspl[0]]) is float:
                    typeval='float'
                    self._keywd[kwspl[0]] = float(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is list:
                    typeval='list'
                    self._keywd[kwspl[0]].append(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is bool:
                    if kwspl[1] in ['True', 'False']:
                        typeval='boolean'
                        self._keywd[kwspl[0]] = kwspl[1] in ['True']
                    else:
                        typeval='string'
                        self._keywd[kwspl[0]] = kwspl[1]
                        msgs.warn(kwspl[0]+" should be of type boolean (True/False)", verbose=self._verbose)
                elif self._keywd[kwspl[0]] is None:
                    typeval='None'
                    self._keywd[kwspl[0]] = None
                else:
                    msgs.error("I don't understand the format on line:"+msgs.newline()+self._idstr+"   "+instr)
            self._keych[kwspl[0]] = 0 # Set keych for this keyword to zero to show that this has been changed
        # Check that all required keywords were changed
        for i in range(len(keywdk)):
            if self._keych[keywdk[i]] == 1: msgs.error(keywdk[i]+" must be set for -"+msgs.newline()+self._idstr+"   "+instr)
        # Append the final set of keywords
        mp['mkey'].append(self._keywd.copy())
        return mp, parid

    def parin(self, i, par):
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
        return pin


    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, getinfl=False, ddpid=None, getstdd=None):
        """
        Return the parameters for a Gaussian function to be used by 'call'
        The only thing that should be changed here is the parb values
        """
        levadd=0
        params=np.zeros(self._pnumr)
        parinf=[]
        for i in range(self._pnumr):
            lnkprm = None
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
                params[i] = self.parin(i, p[getid])
                if mp['mfix'][ival][i] == 0: parinf.append(getid) # If parameter not fixed, append it to the influence array
            else:
                params[i] = lnkprm
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if params[0] == 0: return params, 1
            if nexbin[0] == "km/s": msgs.error("bintype is set to 'km/s', when shift is specified in Angstroms.")
            elif nexbin[0] == "A" : return params, int(round(2.0*np.sqrt(2.0*np.log(2.0))*nexbin[1]/params[0] + 0.5))
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getstdd is not None:
            fact = 2.0*np.sqrt(2.0*np.log(2.0))
            return getstdd[1]*(1.0+getstdd[0]*params[0]/(fact*299792.458)), getstdd[2]*(1.0-getstdd[0]*params[0]/(fact*299792.458))
        elif getinfl: return params, parinf
        else: return params





class vshift(alfunc_base.Base) :
    """
    Shifts the model spectrum by a velocity p[0]:
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'vshift'			# ID string for this class
        self._pnumr   = 1				# Total number of parameters fed in
        self._keywd   = dict({'blind':False})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'blind':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'blind':""})			# Require keywd to be changed (1 for yes, 0 for no)
        self._parid   = ['value']		# Name of each parameter
        self._defpar  = [ 0.0 ]			# Default values for parameters that are not provided
        self._fixpar  = [ None ]		# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ] ]	# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0] ]	# What should these limiting values be
        self._svfmt   = [ "{0:.7g}" ]	# Specify the format used to print or save output
        self._prekw   = []				# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        if getinst: return

    def call_CPU(self, x, p, ncpus=1, mkey=None):
        """
        Define the functional form of the model
        --------------------------------------------------------
        x  : array of wavelengths
        y  : model flux array
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        return x / (1.0 + p[0]/299792.458)


    def load(self, instr, cntr, mp, specid, forcefix=False):
        """
        Load the parameters in the input model file
        --------------------------------------------------------
        instr: input string for the parameters and keywords of
               model to be loaded (ignoring the identifier at
               the beginning of the model line).
        cntr : The line number of the model (e.g. if it's the
               first model line, cntr=0)
        mp   : modpass --> A dictionary with the details of all
               models read in so far.
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.
        --------------------------------------------------------
        """
        def check_tied_param(ival, cntr, mps, iind):
            havtie = False
            tieval=ival.lstrip('+-.0123456789')
            if tieval[0:2] in ['E+', 'e+', 'E-', 'e-']: # Scientific Notation is used.
                tieval=tieval[2:].lstrip('.0123456789')
            inval=float(ival.rstrip(tieval))
            if len(tieval) == 0: # Parameter is not tied
                mps['mtie'][cntr].append(-1)
                if forcefix:
                    mps['mfix'][cntr].append(1)
                else:
                    mps['mfix'][cntr].append(0)
            else: # parameter is tied
                # Determine if this parameter is fixed
                if tieval[0].isupper() or forcefix: mps['mfix'][cntr].append(1)
                else: mps['mfix'][cntr].append(0)
                # Check if this tieval has been used before
                if len(mps['tpar']) == 0: # If it's the first known tied parameter in the model
                    mps['tpar'].append([])
                    mps['tpar'][0].append(tieval)
                    mps['tpar'][0].append(len(mps['p0']))
                    mps['mtie'][cntr].append(-1) # i.e. not tied to anything
                else:
                    for j in range(0,len(mps['tpar'])):
                        if mps['tpar'][j][0] == tieval:
                            mps['mtie'][cntr].append(j)
                            havtie = True
                    if havtie == False: # create a New tied parameter
                        mps['tpar'].append([])
                        mps['tpar'][-1].append(tieval)
                        mps['tpar'][-1].append(len(mps['p0']))
                        mps['mtie'][cntr].append(-1) # i.e. not tied to anything
            if havtie == False: mps['p0'].append(inval)
            mps['mpar'][cntr].append(inval)
            mps['mlim'][cntr].append([self._limits[iind][i] if self._limited[iind][i]==1 else None for i in range(2)])
            return mps
        ################
        isspl=instr.split()
        # Seperate the parameters from the keywords
        kywrd = []
        keywdk = list(self._keywd.keys())
        keywdk[:] = (kych for kych in keywdk if kych[:] != 'input') # Remove the keyword 'input'
        param = [None for all in range(self._pnumr)]
        parid = [i for i in range(self._pnumr)]
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                kwspl = isspl[i].split('=')
                if kwspl[0] in keywdk:
                    self._keywd['input'][kwspl[0]]=1
                    kywrd.append(isspl[i])
                else: msgs.error("Keyword '"+isspl[i]+"' is unknown for -"+msgs.newline()+self._idstr+"   "+instr)
            else:
                param[i] = isspl[i]
                self._keywd['input'][self._parid[i]]=1
        # Do some quick checks
        if len(param) != self._pnumr:
            msgs.error("Incorrect number of parameters (should be "+str(self._pnumr)+"):"+msgs.newline()+self._idstr+"   "+instr)
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(self._pnumr):
            mp = check_tied_param(param[i], cntr, mp, i)
        # Now load the keywords:
        for i in range(len(kywrd)):
            kwspl = kywrd[i].split('=')
            ksspl = kwspl[1].split(',')
            for j in range(len(ksspl)):
                if type(self._keywd[kwspl[0]]) is int:
                    typeval='integer'
                    self._keywd[kwspl[0]] = int(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is str:
                    typeval='string'
                    self._keywd[kwspl[0]] = kwspl[1]
                elif type(self._keywd[kwspl[0]]) is float:
                    typeval='float'
                    self._keywd[kwspl[0]] = float(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is list:
                    typeval='list'
                    self._keywd[kwspl[0]].append(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is bool:
                    if kwspl[1] in ['True', 'False']:
                        typeval='boolean'
                        self._keywd[kwspl[0]] = kwspl[1] in ['True']
                    else:
                        typeval='string'
                        self._keywd[kwspl[0]] = kwspl[1]
                        msgs.warn(kwspl[0]+" should be of type boolean (True/False)", verbose=self._verbose)
                elif self._keywd[kwspl[0]] is None:
                    typeval='None'
                    self._keywd[kwspl[0]] = None
                else:
                    msgs.error("I don't understand the format on line:"+msgs.newline()+self._idstr+"   "+instr)
            self._keych[kwspl[0]] = 0 # Set keych for this keyword to zero to show that this has been changed
        # Check that all required keywords were changed
        for i in range(len(keywdk)):
            if self._keych[keywdk[i]] == 1: msgs.error(keywdk[i]+" must be set for -"+msgs.newline()+self._idstr+"   "+instr)
        # Append the final set of keywords
        mp['mkey'].append(self._keywd.copy())
        return mp, parid

    def parin(self, i, par):
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
        return pin

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, getinfl=False, ddpid=None, getstdd=None):
        """
        Return the parameters for a Gaussian function to be used by 'call'
        The only thing that should be changed here is the parb values
        """
        levadd=0
        params=np.zeros(self._pnumr)
        parinf=[]
        for i in range(self._pnumr):
            lnkprm = None
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
                params[i] = self.parin(i, p[getid])
                if mp['mfix'][ival][i] == 0: parinf.append(getid) # If parameter not fixed, append it to the influence array
            else:
                params[i] = lnkprm
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if params[0] == 0: return params, 1
            if nexbin[0] == "km/s": return params, int(round(2.0*np.sqrt(2.0*np.log(2.0))*nexbin[1]/params[0] + 0.5))
            elif nexbin[0] == "A" : msgs.error("bintype is set to 'A' for Angstroms, when shift is specified as a velocity.")
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getstdd is not None:
            fact = 2.0*np.sqrt(2.0*np.log(2.0))
            return getstdd[1]*(1.0+getstdd[0]*params[0]/(fact*299792.458)), getstdd[2]*(1.0-getstdd[0]*params[0]/(fact*299792.458))
        elif getinfl: return params, parinf
        else: return params


class vshiftscale(alfunc_base.Base) :
    """
    Shifts the model spectrum by a velocity p[0], and scales by p[1]:
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'vshiftscale'			# ID string for this class
        self._pnumr   = 2				# Total number of parameters fed in
        self._keywd   = dict({'blind':False})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'blind':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'blind':""})			# Require keywd to be changed (1 for yes, 0 for no)
        self._parid   = ['shift', 'scale']		# Name of each parameter
        self._defpar  = [ 0.0, 0.0 ]			# Default values for parameters that are not provided
        self._fixpar  = [ None, None ]		# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ] , [1  ,1  ] ]	# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0], [0.5,2.0] ]	# What should these limiting values be
        self._svfmt   = [ "{0:.7g}", "{0:.7g}" ]	# Specify the format used to print or save output
        self._prekw   = []				# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        if getinst: return

    def call_CPU(self, x, p, ncpus=1, mkey=None):
        """
        Define the functional form of the model
        --------------------------------------------------------
        x  : array of wavelengths
        y  : model flux array
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        # Scale the wavelength array relative to the reference value
        wref = x[x.size//2]
        xsc = wref + (x-wref)*p[1]
        # Shift the wavelengths by a velocity p[0]
        return xsc / (1.0 + p[0]/299792.458)

    def load(self, instr, cntr, mp, specid, forcefix=False):
        """
        Load the parameters in the input model file
        --------------------------------------------------------
        instr: input string for the parameters and keywords of
               model to be loaded (ignoring the identifier at
               the beginning of the model line).
        cntr : The line number of the model (e.g. if it's the
               first model line, cntr=0)
        mp   : modpass --> A dictionary with the details of all
               models read in so far.
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.
        --------------------------------------------------------
        """
        def check_tied_param(ival, cntr, mps, iind):
            havtie = False
            tieval=ival.lstrip('+-.0123456789')
            if tieval[0:2] in ['E+', 'e+', 'E-', 'e-']: # Scientific Notation is used.
                tieval=tieval[2:].lstrip('.0123456789')
            inval=float(ival.rstrip(tieval))
            if len(tieval) == 0: # Parameter is not tied
                mps['mtie'][cntr].append(-1)
                if forcefix:
                    mps['mfix'][cntr].append(1)
                else:
                    mps['mfix'][cntr].append(0)
            else: # parameter is tied
                # Determine if this parameter is fixed
                if tieval[0].isupper() or forcefix: mps['mfix'][cntr].append(1)
                else: mps['mfix'][cntr].append(0)
                # Check if this tieval has been used before
                if len(mps['tpar']) == 0: # If it's the first known tied parameter in the model
                    mps['tpar'].append([])
                    mps['tpar'][0].append(tieval)
                    mps['tpar'][0].append(len(mps['p0']))
                    mps['mtie'][cntr].append(-1) # i.e. not tied to anything
                else:
                    for j in range(0,len(mps['tpar'])):
                        if mps['tpar'][j][0] == tieval:
                            mps['mtie'][cntr].append(j)
                            havtie = True
                    if havtie == False: # create a New tied parameter
                        mps['tpar'].append([])
                        mps['tpar'][-1].append(tieval)
                        mps['tpar'][-1].append(len(mps['p0']))
                        mps['mtie'][cntr].append(-1) # i.e. not tied to anything
            if havtie == False: mps['p0'].append(inval)
            mps['mpar'][cntr].append(inval)
            mps['mlim'][cntr].append([self._limits[iind][i] if self._limited[iind][i]==1 else None for i in range(2)])
            return mps
        ################
        isspl=instr.split(",")
        # Seperate the parameters from the keywords
        kywrd = []
        keywdk = list(self._keywd.keys())
        keywdk[:] = (kych for kych in keywdk if kych[:] != 'input') # Remove the keyword 'input'
        param = [None for all in range(self._pnumr)]
        parid = [i for i in range(self._pnumr)]
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                kwspl = isspl[i].split('=')
                if kwspl[0] in keywdk:
                    self._keywd['input'][kwspl[0]]=1
                    kywrd.append(isspl[i])
                else: msgs.error("Keyword '"+isspl[i]+"' is unknown for -"+msgs.newline()+self._idstr+"   "+instr)
            else:
                param[i] = isspl[i]
                self._keywd['input'][self._parid[i]]=1
        # Do some quick checks
        if len(param) != self._pnumr:
            msgs.error("Incorrect number of parameters (should be "+str(self._pnumr)+"):"+msgs.newline()+self._idstr+"   "+instr)
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(self._pnumr):
            mp = check_tied_param(param[i], cntr, mp, i)
        # Now load the keywords:
        for i in range(len(kywrd)):
            kwspl = kywrd[i].split('=')
            ksspl = kwspl[1].split(',')
            for j in range(len(ksspl)):
                if type(self._keywd[kwspl[0]]) is int:
                    typeval='integer'
                    self._keywd[kwspl[0]] = int(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is str:
                    typeval='string'
                    self._keywd[kwspl[0]] = kwspl[1]
                elif type(self._keywd[kwspl[0]]) is float:
                    typeval='float'
                    self._keywd[kwspl[0]] = float(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is list:
                    typeval='list'
                    self._keywd[kwspl[0]].append(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is bool:
                    if kwspl[1] in ['True', 'False']:
                        typeval='boolean'
                        self._keywd[kwspl[0]] = kwspl[1] in ['True']
                    else:
                        typeval='string'
                        self._keywd[kwspl[0]] = kwspl[1]
                        msgs.warn(kwspl[0]+" should be of type boolean (True/False)", verbose=self._verbose)
                elif self._keywd[kwspl[0]] is None:
                    typeval='None'
                    self._keywd[kwspl[0]] = None
                else:
                    msgs.error("I don't understand the format on line:"+msgs.newline()+self._idstr+"   "+instr)
            self._keych[kwspl[0]] = 0 # Set keych for this keyword to zero to show that this has been changed
        # Check that all required keywords were changed
        for i in range(len(keywdk)):
            if self._keych[keywdk[i]] == 1: msgs.error(keywdk[i]+" must be set for -"+msgs.newline()+self._idstr+"   "+instr)
        # Append the final set of keywords
        mp['mkey'].append(self._keywd.copy())
        return mp, parid

    def parin(self, i, par):
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
        return pin

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, getinfl=False, ddpid=None, getstdd=None):
        """
        Return the parameters for a Gaussian function to be used by 'call'
        The only thing that should be changed here is the parb values
        """
        levadd=0
        params=np.zeros(self._pnumr)
        parinf=[]
        for i in range(self._pnumr):
            lnkprm = None
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
                params[i] = self.parin(i, p[getid])
                if mp['mfix'][ival][i] == 0: parinf.append(getid) # If parameter not fixed, append it to the influence array
            else:
                params[i] = lnkprm
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if params[0] == 0: return params, 1
            if nexbin[0] == "km/s": return params, int(round(2.0*np.sqrt(2.0*np.log(2.0))*nexbin[1]/params[0] + 0.5))
            elif nexbin[0] == "A" : msgs.error("bintype is set to 'A' for Angstroms, when shift is specified as a velocity.")
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getstdd is not None:
            fact = 2.0*np.sqrt(2.0*np.log(2.0))
            return getstdd[1]*(1.0+getstdd[0]*params[0]/(fact*299792.458)), getstdd[2]*(1.0-getstdd[0]*params[0]/(fact*299792.458))
        elif getinfl: return params, parinf
        else: return params


class polyshift(alfunc_base.Base) :
    """
    Shifts the model spectrum by a velocity p[0], and scales by p[1], p[2], p[3]... assuming a polynomial:
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'polyshift'			# ID string for this class
        self._pnumr   = 2				# Total number of parameters fed in
        self._keywd   = dict({'blind':False})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'blind':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'blind':""})			# Require keywd to be changed (1 for yes, 0 for no)
        self._parid   = ['shift', 'scale']		# Name of each parameter
        self._defpar  = [ 0.0, 0.0 ]			# Default values for parameters that are not provided
        self._fixpar  = [ None, None ]		# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ] , [1  ,1  ] ]	# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0], [0.5,2.0] ]	# What should these limiting values be
        self._svfmt   = [ "{0:.7g}", "{0:.7g}" ]	# Specify the format used to print or save output
        self._prekw   = []				# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        if getinst: return

    def call_CPU(self, x, p, ncpus=1, mkey=None):
        """
        Define the functional form of the model
        --------------------------------------------------------
        x  : array of wavelengths
        y  : model flux array
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        # Scale the wavelength array relative to the reference value
        wref = x[x.size//2]
        xsc = wref.copy()
        for pp in range(1, p.size):
            xsc += p[pp] * (x-wref)**pp
        # Shift the wavelengths by a velocity p[0]
        return xsc / (1.0 + p[0]/299792.458)

    def load(self, instr, cntr, mp, specid, forcefix=False):
        """
        Load the parameters in the input model file
        --------------------------------------------------------
        instr: input string for the parameters and keywords of
               model to be loaded (ignoring the identifier at
               the beginning of the model line).
        cntr : The line number of the model (e.g. if it's the
               first model line, cntr=0)
        mp   : modpass --> A dictionary with the details of all
               models read in so far.
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.
        --------------------------------------------------------
        """
        def check_tied_param(ival, cntr, mps, iind):
            havtie = False
            tieval=ival.lstrip('+-.0123456789')
            if tieval[0:2] in ['E+', 'e+', 'E-', 'e-']: # Scientific Notation is used.
                tieval=tieval[2:].lstrip('.0123456789')
            inval=float(ival.rstrip(tieval))
            if len(tieval) == 0: # Parameter is not tied
                mps['mtie'][cntr].append(-1)
                if forcefix:
                    mps['mfix'][cntr].append(1)
                else:
                    mps['mfix'][cntr].append(0)
            else: # parameter is tied
                # Determine if this parameter is fixed
                if tieval[0].isupper() or forcefix: mps['mfix'][cntr].append(1)
                else: mps['mfix'][cntr].append(0)
                # Check if this tieval has been used before
                if len(mps['tpar']) == 0: # If it's the first known tied parameter in the model
                    mps['tpar'].append([])
                    mps['tpar'][0].append(tieval)
                    mps['tpar'][0].append(len(mps['p0']))
                    mps['mtie'][cntr].append(-1) # i.e. not tied to anything
                else:
                    for j in range(0,len(mps['tpar'])):
                        if mps['tpar'][j][0] == tieval:
                            mps['mtie'][cntr].append(j)
                            havtie = True
                    if havtie == False: # create a New tied parameter
                        mps['tpar'].append([])
                        mps['tpar'][-1].append(tieval)
                        mps['tpar'][-1].append(len(mps['p0']))
                        mps['mtie'][cntr].append(-1) # i.e. not tied to anything
            if havtie == False: mps['p0'].append(inval)
            mps['mpar'][cntr].append(inval)
            if iind >= self._pnumr:
                # mps['mlim'][cntr].append([self._limits[1][i] if self._limited[1][i]==1 else None for i in range(2)])
                mps['mlim'][cntr].append([None for i in range(2)])
            else:
                mps['mlim'][cntr].append([self._limits[iind][i] if self._limited[iind][i]==1 else None for i in range(2)])
            return mps
        ################
        isspl=instr.split(",")
        pnumr = 0
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                if isspl[i].split('=')[0] == self._parid[1]: pnumr += 1
            else: pnumr += 1
        # Seperate the parameters from the keywords
        kywrd = []
        keywdk = list(self._keywd.keys())
        keywdk[:] = (kych for kych in keywdk if kych[:] != 'input') # Remove the keyword 'input'
        param = [None for all in range(pnumr)]
        parid = [i for i in range(pnumr)]
        for pp in range(1,pnumr): parid[pp] = 1
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                kwspl = isspl[i].split('=')
                if kwspl[0] in keywdk:
                    self._keywd['input'][kwspl[0]]=1
                    kywrd.append(isspl[i])
                else: msgs.error("Keyword '"+isspl[i]+"' is unknown for -"+msgs.newline()+self._idstr+"   "+instr)
            else:
                param[i] = isspl[i]
                self._keywd['input'][parid[i]]=1
        # Do some quick checks
        if len(param) != pnumr:
            msgs.error("Incorrect number of parameters (should be "+str(pnumr)+"):"+msgs.newline()+self._idstr+"   "+instr)
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(pnumr):
            mp = check_tied_param(param[i], cntr, mp, i)
        # Now load the keywords:
        for i in range(len(kywrd)):
            kwspl = kywrd[i].split('=')
            ksspl = kwspl[1].split(',')
            for j in range(len(ksspl)):
                if type(self._keywd[kwspl[0]]) is int:
                    typeval='integer'
                    self._keywd[kwspl[0]] = int(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is str:
                    typeval='string'
                    self._keywd[kwspl[0]] = kwspl[1]
                elif type(self._keywd[kwspl[0]]) is float:
                    typeval='float'
                    self._keywd[kwspl[0]] = float(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is list:
                    typeval='list'
                    self._keywd[kwspl[0]].append(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is bool:
                    if kwspl[1] in ['True', 'False']:
                        typeval='boolean'
                        self._keywd[kwspl[0]] = kwspl[1] in ['True']
                    else:
                        typeval='string'
                        self._keywd[kwspl[0]] = kwspl[1]
                        msgs.warn(kwspl[0]+" should be of type boolean (True/False)", verbose=self._verbose)
                elif self._keywd[kwspl[0]] is None:
                    typeval='None'
                    self._keywd[kwspl[0]] = None
                else:
                    msgs.error("I don't understand the format on line:"+msgs.newline()+self._idstr+"   "+instr)
            self._keych[kwspl[0]] = 0 # Set keych for this keyword to zero to show that this has been changed
        # Check that all required keywords were changed
        for i in range(len(keywdk)):
            if self._keych[keywdk[i]] == 1: msgs.error(keywdk[i]+" must be set for -"+msgs.newline()+self._idstr+"   "+instr)
        # Append the final set of keywords
        mp['mkey'].append(self._keywd.copy())
        return mp, parid

    def parin(self, i, par):
        """
        This routine converts a parameter in the input model file
        to the parameter used in 'call'
        --------------------------------------------------------
        When writing a new function, one should change how each
        input parameter 'par' is converted into a parameter used
        in the function specified by 'call'
        --------------------------------------------------------
        """
        return par

    def set_pinfo(self, pinfo, level, mp, lnk, mnum):
        """
        Place limits on the functions parameters (as specified in init)
        Nothing should be changed here.
        """
        pnumr = len(mp['mpar'][mnum])
        add = pnumr
        levadd = 0
        for i in range(pnumr):
            # First Check if there are any operations/links to perform on this parameter
            if len(lnk['opA']) != 0:
                breakit=False
                for j in range(len(mp['tpar'])):
                    if breakit: break
                    if mp['tpar'][j][1] == level+levadd: # then this is a tied parameter
                        # Check if this tied parameter is to be linked to another parameter
                        for k in range(len(lnk['opA'])):
                            if mp['tpar'][j][0] == lnk['opA'][k]:
                                ttext = lnk['exp'][k]
                                for l in lnk['opB'][k]:
                                    for m in range(len(mp['tpar'])):
                                        if mp['tpar'][m][0] == l:
                                            pn = mp['tpar'][m][1]
                                            break
                                    ttext = ttext.replace(l,'p[{0:d}]'.format(pn))
                                pinfo[level+levadd]['tied'] = ttext
                                breakit = True
                                break
            # Now set limits and fixed values
            if mp['mtie'][mnum][i] >= 0:
                add -= 1
            elif mp['mtie'][mnum][i] <= -2:
                pinfo[level+levadd]['limited'] = [0 if j is None else 1 for j in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['limits']  = [0.0 if j is None else float(j) for j in mp['mlim'][mnum][i]]
                mp['mfix'][mnum][i] = -1
                levadd += 1
            else:
                pinfo[level+levadd]['limited'] = [0 if j is None else 1 for j in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['limits']  = [0.0 if j is None else float(j) for j in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['fixed']   = mp['mfix'][mnum][i]
                levadd += 1
        return pinfo, add

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, getinfl=False, ddpid=None, getstdd=None):
        """
        Return the parameters for a Gaussian function to be used by 'call'
        The only thing that should be changed here is the parb values
        """
        levadd=0
        pnumr = len(mp['mpar'][ival])
        params=np.zeros(pnumr)
        parinf=[]
        for i in range(pnumr):
            lnkprm = None
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
                params[i] = self.parin(i, p[getid])
                if mp['mfix'][ival][i] == 0: parinf.append(getid) # If parameter not fixed, append it to the influence array
            else:
                params[i] = lnkprm
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if params[0] == 0: return params, 1
            if nexbin[0] == "km/s": return params, int(round(2.0*np.sqrt(2.0*np.log(2.0))*nexbin[1]/params[0] + 0.5))
            elif nexbin[0] == "A" : msgs.error("bintype is set to 'A' for Angstroms, when shift is specified as a velocity.")
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getstdd is not None:
            fact = 2.0*np.sqrt(2.0*np.log(2.0))
            return getstdd[1]*(1.0+getstdd[0]*params[0]/(fact*299792.458)), getstdd[2]*(1.0-getstdd[0]*params[0]/(fact*299792.458))
        elif getinfl: return params, parinf
        else: return params

    def parout(self, params, mp, istart, level, errs=None, reletter=False, conv=None):
        """
        Convert the parameter list to the input form for
        printing to screen or output to a file.
        --------------------------------------------------------
        pval     : List of all parameters
        mp       : modpass, which includes details of all models
        istart   : Model number
        errs     : List of all parameter errors.
        reletter : Set to true if you want to reletter the tied
                   and fixed parameters (a will be used for the
                   first parameter, b is used for the second...)
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.
        --------------------------------------------------------
        """
        if errs is None: errors = params
        else: errors = errs
        numpar = len(mp['mpar'][istart])
        parid = [i for i in range(numpar)]
        for pp in range(1, numpar): parid[pp] = 1
        add = numpar
        havtie = 0
        tienum = 0
        levadd = 0
        outstring = ['  %s ' % (self._idstr)]
        errstring = ['# %s ' % (self._idstr)]

        # Check if we are blinding any parameters with an offset value
        blindoffset = 0
        if 'blindrange' in mp['mkey'][istart]:
            if len(mp['mkey'][istart]['blindrange']) == 2:
                if 'blindseed' in mp['mkey'][istart]:
                    np.random.seed(mp['mkey'][istart]['blindseed'])
                else:
                    np.random.seed(0)
                blindoffset = np.random.uniform(int(mp['mkey'][istart]['blindrange'][0]), int(mp['mkey'][istart]['blindrange'][1]))
            # Reset the seed
            self.resetseed()

        for i in range(numpar):
            ipid = i
            if i >= 3: ipid = 3
            if mp['mkey'][istart]['input'][parid[ipid]] == 0: # Parameter not given as input
                outstring.append( "" )
                errstring.append( "" )
                continue
            elif mp['mkey'][istart]['input'][parid[ipid]] == 1: pretxt = ""   # Parameter is given as input, without parid
            else: pretxt = parid[ipid]+"="                                    # Parameter is given as input, with parid
            if mp['mtie'][istart][i] >= 0:
                if reletter:
                    newfmt=pretxt+self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],self._svfmt[parid[ipid]]+'{1:c}')
                    outstring.append( (newfmt).format(blindoffset+params[mp['tpar'][mp['mtie'][istart][i]][1]],97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                    if conv is None:
                        errstring.append( (newfmt).format(errors[mp['tpar'][mp['mtie'][istart][i]][1]],97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                    else:
                        if params[mp['tpar'][mp['mtie'][istart][i]][1]] < conv: cvtxt = "CONVERGED"
                        else: cvtxt = "!!!!!!!!!"
                        errstring.append( ('--{0:s}--{1:c}    ').format(cvtxt,97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                else:
                    newfmt=pretxt+self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],self._svfmt[parid[ipid]]+'{1:s}')
                    outstring.append( (newfmt).format(blindoffset+params[mp['tpar'][mp['mtie'][istart][i]][1]],mp['tpar'][mp['mtie'][istart][i]][0]) )
                    if conv is None:
                        errstring.append( (newfmt).format(errors[mp['tpar'][mp['mtie'][istart][i]][1]],mp['tpar'][mp['mtie'][istart][i]][0]) )
                    else:
                        if params[mp['tpar'][mp['mtie'][istart][i]][1]] < conv: cvtxt = "CONVERGED"
                        else: cvtxt = "!!!!!!!!!"
                        errstring.append( ('--{0:s}--{1:s}    ').format(cvtxt,mp['tpar'][mp['mtie'][istart][i]][0]) )
                add -= 1
            else:
                if havtie != 2:
                    if havtie == 0: # First searching for the very first instance of a tied parameter
                        for tn in range(0,len(mp['tpar'])):
                            if mp['tpar'][tn][1] == level+levadd:
                                tienum=tn
                                havtie=1
                                break
                    if mp['tpar'][tienum][1] == level+levadd:
                        if reletter:
                            newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[parid[i]]+'{1:c}')
                            outstring.append( (newfmt).format(blindoffset+params[level+levadd],97+tienum-32*mp['mfix'][istart][1]) )
                            if conv is None:
                                errstring.append( (newfmt).format(errors[level+levadd],97+tienum-32*mp['mfix'][istart][1]) )
                            else:
                                if params[level+levadd] < conv: cvtxt = "CONVERGED"
                                else: cvtxt = "!!!!!!!!!"
                                errstring.append( ('--{0:s}--{1:c}    ').format(cvtxt,97+tienum-32*mp['mfix'][istart][1]) )
                        else:
                            newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[parid[i]]+'{1:s}')
                            outstring.append( (newfmt).format(blindoffset+params[level+levadd],mp['tpar'][tienum][0]) )
                            if conv is None:
                                errstring.append( (newfmt).format(errors[level+levadd],mp['tpar'][tienum][0]) )
                            else:
                                if params[level+levadd] < conv: cvtxt = "CONVERGED"
                                else: cvtxt = "!!!!!!!!!"
                                errstring.append( ('--{0:s}--{1:s}    ').format(cvtxt,mp['tpar'][tienum][0]) )
                        tienum += 1
                        if tienum == len(mp['tpar']): havtie = 2 # Stop searching for 1st instance of tied param
                    else:
                        newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[parid[i]])
                        outstring.append( (newfmt).format(blindoffset+params[level+levadd]) )
                        if conv is None:
                            errstring.append( (newfmt).format(errors[level+levadd]) )
                        else:
                            if params[level+levadd] < conv: cvtxt = "CONVERGED"
                            else: cvtxt = "!!!!!!!!!"
                            errstring.append( ('--{0:s}--    ').format(cvtxt) )
                else:
                    newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[parid[i]])
                    outstring.append( (newfmt).format(blindoffset+params[level+levadd]) )
                    if conv is None:
                        errstring.append( (newfmt).format(errors[level+levadd]) )
                    else:
                        if params[level+levadd] < conv: cvtxt = "CONVERGED"
                        else: cvtxt = "!!!!!!!!!"
                        errstring.append( ('--{0:s}--    ').format(cvtxt) )
                levadd += 1
        level += add
        # Now write in the keywords
        keys = list(mp['mkey'][istart].keys())
        keys[:] = (kych for kych in keys if kych[:] != 'input') # Remove the keyword 'input'
        for i in range(len(keys)):
            if mp['mkey'][istart]['input'][keys[i]] == 0: # This keyword wasn't provided as user input
                outstring.append( "" )
                errstring.append( "" )
                continue
            if type(mp['mkey'][istart][keys[i]]) is list: outkw = ','.join(map(str,mp['mkey'][istart][keys[i]]))
            else: outkw = mp['mkey'][istart][keys[i]]
            if self._keyfm[keys[i]] != "":
                outstring.append( ('{0:s}='+self._keyfm[keys[i]]).format(keys[i],outkw) )
                errstring.append( ('{0:s}='+self._keyfm[keys[i]]).format(keys[i],outkw) )
            else:
                outstring.append( '%s=%s' % (keys[i],outkw) )
                errstring.append( '%s=%s' % (keys[i],outkw) )
        # Now place the keywords specified in self._prekw at the beginning of the return string:
        if len(self._prekw) != 0:
            insind = 1
            for i in range(len(self._prekw)):
                delind = -1
                for j in range(len(keys)):
                    if self._prekw[i] == keys[j]:
                        delind = numpar+insind+j
                        del keys[j]
                        break
                if delind == -1: msgs.bug("prekw variable for function "+self._idstr+" contains bad argument", verbose=self._verbose)
                outstring.insert(insind,outstring[delind])
                errstring.insert(insind,errstring[delind])
                del outstring[delind+1]
                del errstring[delind+1]
                insind += 1
        if mp['mkey'][istart]['blind'] and conv is None:
            retout = "       ------ BLIND MODEL ------\n"
            #reterr = "       ------ BLIND MODEL ------\n"
            reterr = '  '.join(errstring) + '\n'
        else:
            retout = '  '.join(outstring) + '\n'
            reterr = '  '.join(errstring) + '\n'
        # Return the strings and the new level
        if errs is not None or conv is not None:
            return retout, reterr, level
        else:
            return retout, level
