import numpy as np
from alis import almsgs
from alis import alfunc_base
msgs=almsgs.msgs()

class Variable(alfunc_base.Base):
    """
    Returns a single variable:
    p[0] = value
    This is useful when you want to calculate the value
    and error on a combination of other model parameters.
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'variable'								# ID string for this class
        self._pnumr   = 1										# Total number of parameters fed in
        self._keywd   = dict({'specid':[], 'continuum':False, 'blind':False, 'step':0.0})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'specid':0,  'continuum':0,     'blind':0,     'step':0})		# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'specid':"", 'continuum':"",    'blind':"",    'step':""})	# Format for the keyword. "" is the Default setting
        self._parid   = ['value']								# Name of each parameter
        self._defpar  = [ 0.0 ]									# Default values for parameters that are not provided
        self._fixpar  = [ None ]								# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ] ]							# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0] ]							# What should these limiting values be
        self._svfmt   = [ "{0:.8g}" ]							# Specify the format used to print or save output
        self._prekw   = []										# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        if getinst: return

    def call_CPU(self, x, p, ae='va', mkey=None, ncpus=1):
        """
        Define the functional form of the model for the CPU
        --------------------------------------------------------
        x  : array of wavelengths
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        return


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
        return pin


    def set_pinfo(self, pinfo, level, mp, lnk, mnum):
        """
        Place limits on the functions parameters (as specified in init)
        Nothing should be changed here.
        """
        add = self._pnumr
        levadd = 0
        for i in range(self._pnumr):
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
            if mp['mtie'][mnum][i] >= 0: add -= 1
            elif mp['mtie'][mnum][i] <= -2:
                pinfo[level+levadd]['step'] = mp['mkey'][mnum]['step']
                pinfo[level+levadd]['limited'] = [0 if o is None else 1 for o in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['limits']  = [0.0 if o is None else float(o) for o in mp['mlim'][mnum][i]]
                mp['mfix'][mnum][i] = -1
                levadd += 1
            else:
                pinfo[level+levadd]['step'] = mp['mkey'][mnum]['step']
                pinfo[level+levadd]['limited'] = [0 if o is None else 1 for o in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['limits']  = [0.0 if o is None else float(o) for o in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['fixed']   = mp['mfix'][mnum][i]
                levadd += 1
        return pinfo, add


    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, ddpid=None, getinfl=False):
        levadd=0
        params=np.zeros(self._pnumr)
        parinf=[]
        for i in range(self._pnumr):
            parb = None
            if mp['mtie'][ival][i] >= 0:
                getid = mp['tpar'][mp['mtie'][ival][i]][1]
            elif mp['mtie'][ival][i] <= -2:
                msgs.error("You are not allowed to link to the function 'variable'")
            else:
                getid = level+levadd
                levadd+=1
            params[i] = self.parin(i, p[getid], parb)
            if mp['mfix'][ival][i] == 0: parinf.append(getid)
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if params[1] == 0.0: msgs.error("Cannot calculate "+self._idstr+" subpixellation -- width = 0.0")
            if nexbin[0] == "km/s": return params, 1
            elif nexbin[0] == "A" : return params, 1
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getinfl: return params, parinf
        else: return params
