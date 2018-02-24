import numpy as np
from alis import almsgs
from alis import alfunc_base
msgs=almsgs.msgs()

class Polynomial(alfunc_base.Base) :
    """
    Returns a polynomial of the form:
    p[0] = constant
    p[1] = coefficient of the term x
    p[2] = coefficient of the term x**2
    p[3] = coefficient of the term x**3
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
        self._idstr   = 'polynomial'								# ID string for this class
        self._pnumr   = 1											# Total number of parameters fed in
        self._keywd   = dict({'specid':[], 'continuum':False, 'blind':False, 'scale':[]})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'specid':0,  'continuum':0,     'blind':0,     'scale':0})		# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'specid':"", 'continuum':"",    'blind':"",    'scale':""})		# Format for the keyword. "" is the Default setting
        self._parid   = ['coefficient']				# Name of each parameter
        self._defpar  = [ 0.0 ]						# Default values for parameters that are not provided
        self._fixpar  = [ None ]					# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ] ]				# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0] ]				# What should these limiting values be
        self._svfmt   = [ "{0:.8g}" ]				# Specify the format used to print or save output
        self._prekw   = []							# Specify the keywords to print out before the parameters
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
            for m in range(1,len(par)):
                modret += par[m]*(x**m)
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

    def adjust_fix(self, mp, cntr, jval, parj):
        """
        Adjust the parameters in the input model file
        --------------------------------------------------------
        cntr : The line number of the model (e.g. if it's the
               first model line, cntr=0)
        mp   : modpass --> A dictionary with the details of all
               models read in so far.
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.
        --------------------------------------------------------
        """
        # Determine the adjustment that needs to be made.
        value = None
        if self._fixpar[0] not in ['False','false','FALSE']:
            adj = 1
            if self._fixpar[0] not in ['True','true','TRUE']:
                if self._fixpar[0] is True or self._fixpar[0] is False: return mp # This means the default values were used and parameters are fixed
                try:
                    value = float(self._fixpar[0])
                except:
                    msgs.error("Argument of 'fix' only takes True/False/None or float")
        else: adj = 0
        if mp['mfix'][cntr][jval] == adj and value is None: return mp # No adjustment is necessary
        # Determine if the parameter is tied, if it is, store tpnum.
        adjarr=[]
        pradd=0
        if mp['mtie'][cntr][jval] != -1:
            tpnum = mp['mtie'][cntr][jval]
        else:
            tpnum = -1
            for i in range(cntr+1):
                for j in range(len(mp['mtie'][i])):
                    if i == cntr and j == jval:
                        for t in range(len(mp['tpar'])):
                            if mp['tpar'][t][1] == pradd:
                                tpnum = t
                                break
                        break
                    elif mp['mtie'][i][j] == -1: pradd += 1
                if tpnum != -1: break
            if tpnum == -1: # This parameter is not tied to anything else
                mp['mfix'][cntr][jval] = adj
                if value is not None:
                    mp['mpar'][cntr][jval] = value
                    mp['p0'][pradd] = value
                return mp
        # Now tpnum is known, find all associated values
        prnum = mp['tpar'][tpnum][1]
        prsum=-1
        for i in range(len(mp['mtie'])):
            for j in range(len(mp['mtie'][i])):
                if mp['mtie'][i][j] == -1: prsum += 1
                if (prnum == prsum and mp['mtie'][i][j] == -1) or tpnum == mp['mtie'][i][j]: adjarr.append([i,j])
        # Apply the adjustments
        for i in range(len(adjarr)):
            mp['mfix'][adjarr[i][0]][adjarr[i][1]] = adj
            if value is not None:
                mp['mpar'][adjarr[i][0]][adjarr[i][1]] = value
        if value is not None: mp['p0'][prnum] = value
        return mp

    def adjust_lim(self, mp, cntr, jval, jind, parj):
        """
        Adjust the parameters in the input model file
        --------------------------------------------------------
        cntr : The line number of the model (e.g. if it's the
               first model line, cntr=0)
        mp   : modpass --> A dictionary with the details of all
               models read in so far.
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.
        --------------------------------------------------------
        """
        # Determine the adjustment that needs to be made.
        try:
            if self._limited[0][jind] == 0: value = None
            else: value = np.float64(self._limits[0][jind])
        except:
            msgs.error("Argument of 'lim' only takes None or float")
        # Determine if the parameter is tied, if it is, store tpnum.
        adjarr=[]
        pradd=0
        if mp['mtie'][cntr][jval] != -1:
            tpnum = mp['mtie'][cntr][jval]
        else:
            tpnum = -1
            for i in range(cntr+1):
                for j in range(len(mp['mtie'][i])):
                    if i == cntr and j == jval:
                        for t in range(len(mp['tpar'])):
                            if mp['tpar'][t][1] == pradd:
                                tpnum = t
                                break
                        break
                    elif mp['mtie'][i][j] == -1: pradd += 1
                if tpnum != -1: break
            if tpnum == -1: # This parameter is not tied to anything else
                mp['mlim'][cntr][jval][jind] = value
                return mp
        # Now tpnum is known, find all associated values
        prnum = mp['tpar'][tpnum][1]
        prsum=-1
        for i in range(len(mp['mtie'])):
            for j in range(len(mp['mtie'][i])):
                if mp['mtie'][i][j] == -1: prsum += 1
                if (prnum == prsum and mp['mtie'][i][j] == -1) or tpnum == mp['mtie'][i][j]: adjarr.append([i,j])
        # Apply the adjustments
        for i in range(len(adjarr)):
            mp['mlim'][adjarr[i][0]][adjarr[i][1]][jind] = value
        return mp

    def load(self, instr, cntr, mp, specid):
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
                mps['mfix'][cntr].append(0)
            else: # parameter is tied
                # Determine if this parameter is fixed
                if tieval[0].isupper(): mps['mfix'][cntr].append(1)
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
            mps['mlim'][cntr].append([self._limits[0][i] if self._limited[0][i]==1 else None for i in range(2)])
            return mps
        ################
        isspl=instr.split()
        # Seperate the parameters from the keywords
        ptemp, kywrd = [], []
        keywdk = list(self._keywd.keys())
        keywdk[:] = (kych for kych in keywdk if kych[:] != 'input') # Remove the keyword 'input'
        numink = 0
        # Since there's no way of knowing how many coefficients and
        # keywords there are, determine the number of parameters
        pnumr = 0
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                if isspl[i].split('=')[0] == self._parid[0]: pnumr += 1
            else: pnumr += 1
        # Now return to the default code
        param = [None for all in range(pnumr)]
        parid = [i for i in range(pnumr)]
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                kwspl = isspl[i].split('=')
                if kwspl[0] in self._parid:
                    self._keywd['input'][kwspl[0]]=2
                    for j in range(pnumr):
                        if kwspl[0] == self._parid[0]:
                            param[j] = kwspl[1]
                            numink += 1
                            break
                elif kwspl[0] in keywdk:
                    kywrd.append(isspl[i])
                    self._keywd['input'][kwspl[0]]=2
                else: msgs.error("Keyword '"+isspl[i]+"' is unknown for -"+msgs.newline()+self._idstr+"   "+instr)
            else: ptemp.append(isspl[i])
        # Set parameters that weren't given in param
        cntp=0
        fixpid = [0 for all in range(pnumr)]
        for i in range(pnumr):
            if param[i] is None:
                if cntp >= len(ptemp): # Use default values
                    param[i] = str(self._defpar[0])
                    fixpid[i] = 1
                else:
                    param[i] = ptemp[cntp]
                    self._keywd['input'][self._parid[0]]=1
                    cntp += 1
        # Do some quick checks
        if len(ptemp)+numink > pnumr:
            msgs.error("Incorrect number of parameters (should be "+str(pnumr)+"):"+msgs.newline()+self._idstr+"   "+instr)
        if len(specid) > 1: # Force the user to specify a spectrum ID number
            self._keych['specid'] = 1
        specidset=False # Detect is the user has specified specid
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(pnumr):
            mp = check_tied_param(param[i], cntr, mp, i)
        if any(fixpid): # If default values had to be used, fix the values
            for i in range(pnumr):
                if fixpid[i] == 1: mp['mfix'][-1][i] = 1
        # Now load the keywords:
        # Make a copy of the default values
        cpy_keywd = self._keywd.copy()
        cpy_keych = self._keych.copy()
        for i in range(len(kywrd)):
            kwspl = kywrd[i].split('=')
            ksspl = kwspl[1].lstrip('([').rstrip(')]').split(',')
            if kwspl[0] == 'specid':
                sidlist = []
                for j in range(len(ksspl)): # Do some checks
                    if ksspl[j] in sidlist:
                        msgs.error("specid: "+ksspl[j]+" is set twice for -"+msgs.newline()+self._idstr+"   "+instr.replace('\t','  '))
                    else: sidlist.append(ksspl[j])
                    if ksspl[j] not in specid:
                        msgs.error("There is no data with specid: "+ksspl[j]+" for -"+msgs.newline()+self._idstr+"   "+instr.replace('\t','  '))
                specidset=True
            elif kwspl[0] == 'scale':
                scllist = []
                if len(ksspl) != pnumr:
                    msgs.error("Keyword 'scale' for function '{0:s}', must contain the same number".format(self._idstr)+msgs.newline()+"of elements as the number of specified parameters")
                for j in range(len(ksspl)): # Do some checks
                    try:
                        scllist.append(float(ksspl[j]))
                    except:
                        msgs.error("Keyword 'scale' in function '{0:s}' should be an array of floats, not:".format(self._idstr)+msgs.newline()+"{0:s}".format(ksspl[j]))
            for j in range(len(ksspl)):
                if type(cpy_keywd[kwspl[0]]) is int:
                    typeval='integer'
                    cpy_keywd[kwspl[0]] = int(kwspl[1])
                elif type(cpy_keywd[kwspl[0]]) is str:
                    typeval='string'
                    cpy_keywd[kwspl[0]] = kwspl[1]
                elif type(cpy_keywd[kwspl[0]]) is float:
                    typeval='float'
                    cpy_keywd[kwspl[0]] = float(kwspl[1])
                elif type(cpy_keywd[kwspl[0]]) is list and kwspl[0] == 'specid':
                    typeval='list'
                    cpy_keywd[kwspl[0]] = sidlist
                elif type(cpy_keywd[kwspl[0]]) is list and kwspl[0] == 'scale':
                    typeval='list'
                    cpy_keywd[kwspl[0]] = scllist
                elif type(cpy_keywd[kwspl[0]]) is bool:
                    if kwspl[1] in ['True', 'False']:
                        typeval='boolean'
                        cpy_keywd[kwspl[0]] = kwspl[1] in ['True']
                    else:
                        typeval='string'
                        cpy_keywd[kwspl[0]] = kwspl[1]
                        msgs.warn("{0:s} should be of type boolean (True/False)".format(kwspl[0]), verbose=self._verbose)
                elif cpy_keywd[kwspl[0]] is None:
                    typeval='None'
                    cpy_keywd[kwspl[0]] = None
                else:
                    msgs.error("I don't understand the format on line:"+msgs.newline()+self._idstr+"   "+instr)
            cpy_keych[kwspl[0]] = 0 # Set keych for this keyword to zero to show that this has been changed
        # Check that all required keywords were changed
        for i in range(len(keywdk)):
            if cpy_keych[keywdk[i]] == 1: msgs.error(keywdk[i]+" must be set for -"+msgs.newline()+self._idstr+"   "+instr)
        # Check that we have set a specid
        if len(specid) == 1 and not specidset:
            if len(cpy_keywd['specid']) == 0: cpy_keywd['specid'].append(specid[0])
        # Append the final set of keywords
        mp['mkey'].append(cpy_keywd)
        # Before returning, update the parameters of the model
        for i in range(pnumr-1):
            self._parid.append('coefficient')
            self._defpar.append(0.0)
            self._fixpar.append(None)
            self._limited.append([0, 0])
            self._limits.append([0.0,0.0])
        return mp, parid

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
        try:
            pin = parb['ap_1a'][i] * par
        except:
            pin = par
        return pin

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
        if np.all(errs == None):
            errors = params
        else:
            errors = errs
        pnumr = len(mp['mpar'][istart])
        add = pnumr
        havtie = 0
        tienum = 0
        levadd = 0
        outstring = ['  %s ' % (self._idstr)]
        errstring = ['# %s ' % (self._idstr)]
        for i in range(pnumr):
            if mp['mkey'][istart]['input'][self._parid[0]] == 0: # Parameter not given as input
                outstring.append( "" )
                errstring.append( "" )
                continue
            elif mp['mkey'][istart]['input'][self._parid[0]] == 1: pretxt = ""   # Parameter is given as input, without parid
            else: pretxt = self._parid[0]+"="                                    # Parameter is given as input, with parid
            if mp['mtie'][istart][i] >= 0:
                if reletter:
                    newfmt=pretxt+self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],self._svfmt[0]+'{1:c}')
                    outstring.append( (newfmt).format(params[mp['tpar'][mp['mtie'][istart][i]][1]],97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                    if conv is None:
                        errstring.append( (newfmt).format(errors[mp['tpar'][mp['mtie'][istart][i]][1]],97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                    else:
                        if params[mp['tpar'][mp['mtie'][istart][i]][1]] < conv: cvtxt = "CONVERGED"
                        else: cvtxt = "!!!!!!!!!"
                        errstring.append( ('--{0:s}--{1:c}    ').format(cvtxt,97+tienum-32*mp['mfix'][istart][1]) )
                else:
                    newfmt=pretxt+self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],self._svfmt[0]+'{1:s}')
                    outstring.append( (newfmt).format(params[mp['tpar'][mp['mtie'][istart][i]][1]],mp['tpar'][mp['mtie'][istart][i]][0]) )
                    if conv is None:
                        errstring.append( (newfmt).format(errors[mp['tpar'][mp['mtie'][istart][i]][1]],mp['tpar'][mp['mtie'][istart][i]][0]) )
                    else:
                        if params[mp['tpar'][mp['mtie'][istart][i]][1]] < conv: cvtxt = "CONVERGED"
                        else: cvtxt = "!!!!!!!!!"
                        errstring.append( ('--{0:s}--{1:s}    ').format(cvtxt,97+tienum-32*mp['mfix'][istart][1]) )
                add -= 1
            else:
                if havtie != 2:
                    if havtie == 0: # First searching for the very first instance of a tied parameter
                        for tn in range(0,len(mp['tpar'])):
                            if mp['tpar'][tn][1] == level+levadd:
                                tienum=tn
                                havtie=1
                                break
                    if len(mp['tpar']) != 0:
                        if mp['tpar'][tienum][1] == level+levadd:
                            if reletter:
                                newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[0]+'{1:c}')
                                outstring.append( (newfmt).format(params[level+levadd],97+tienum-32*mp['mfix'][istart][1]) )
                                if conv is None:
                                    errstring.append( (newfmt).format(errors[level+levadd],97+tienum-32*mp['mfix'][istart][1]) )
                                else:
                                    if params[level+levadd] < conv: cvtxt = "CONVERGED"
                                    else: cvtxt = "!!!!!!!!!"
                                    errstring.append( ('--{0:s}--{1:c}    ').format(cvtxt,97+tienum-32*mp['mfix'][istart][1]) )
                            else:
                                newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[0]+'{1:s}')
                                outstring.append( (newfmt).format(params[level+levadd],mp['tpar'][tienum][0]) )
                                if conv is None:
                                    errstring.append( (newfmt).format(errors[level+levadd],mp['tpar'][tienum][0]) )
                                else:
                                    if params[level+levadd] < conv: cvtxt = "CONVERGED"
                                    else: cvtxt = "!!!!!!!!!"
                                    errstring.append( ('--{0:s}--{1:s}    ').format(cvtxt,mp['tpar'][tienum][0]) )
                            tienum += 1
                            if tienum == len(mp['tpar']): havtie = 2 # Stop searching for 1st instance of tied param
                        else:
                            newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[0])
                            outstring.append( (newfmt).format(params[level+levadd]) )
                            if conv is None:
                                errstring.append( (newfmt).format(errors[level+levadd]) )
                            else:
                                if params[level+levadd] < conv: cvtxt = "CONVERGED"
                                else: cvtxt = "!!!!!!!!!"
                                errstring.append( ('--{0:s}--    ').format(cvtxt) )
                    else: # There are no tied parameters!
                        newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[0])
                        outstring.append( (newfmt).format(params[level+levadd]) )
                        if conv is None:
                            errstring.append( (newfmt).format(errors[level+levadd]) )
                        else:
                            if params[level+levadd] < conv: cvtxt = "CONVERGED"
                            else: cvtxt = "!!!!!!!!!"
                            errstring.append( ('--{0:s}--    ').format(cvtxt) )
                else:
                    newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[0])
                    outstring.append( (newfmt).format(params[level+levadd]) )
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
            if type(mp['mkey'][istart][keys[i]]) is list: outkw = ','.join(map(str, mp['mkey'][istart][keys[i]]))
            else: outkw = mp['mkey'][istart][keys[i]]
            if self._keyfm[keys[i]] != "":
                outstring.append( ('{0:s}='+self._keyfm[keys[i]]).format(keys[i],outkw) )
                errstring.append( ('{0:s}='+self._keyfm[keys[i]]).format(keys[i],outkw) )
            else:
                outstring.append( '%s=%s' % (keys[i],outkw) )
                errstring.append( '%s=%s' % (keys[i],outkw) )
#                outstring.append( '{0:s}={1:s}'.format(keys[i],outkw) )
#                errstring.append( '{0:s}={1:s}'.format(keys[i],outkw) )
        # Now place the keywords specified in self._prekw at the beginning of the return string:
        if len(self._prekw) != 0:
            insind = 1
            for i in range(len(self._prekw)):
                delind = -1
                for j in range(len(keys)):
                    if self._prekw[i] == keys[j]:
                        delind = pnumr+insind+j
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
            reterr = "       ------ BLIND MODEL ------\n"
        else:
            retout = '  '.join(outstring) + '\n'
            reterr = '  '.join(errstring) + '\n'
        # Return the strings and the new level
        if errs is not None or conv is not None:
            return retout, reterr, level
        else:
            return retout, level

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
            if mp['mtie'][mnum][i] >= 0: add -= 1
            elif mp['mtie'][mnum][i] <= -2:
                pinfo[level+levadd]['limited'] = [0 if j is None else 1 for j in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['limits']  = [0.0 if j is None else np.float64(j) for j in mp['mlim'][mnum][i]]
                mp['mfix'][mnum][i] = -1
                levadd += 1
            else:
                pinfo[level+levadd]['limited'] = [0 if j is None else 1 for j in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['limits']  = [0.0 if j is None else np.float64(j) for j in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['fixed']   = mp['mfix'][mnum][i]
                levadd += 1
        return pinfo, add
            
    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, ddpid=None, getinfl=False):
        """
        Return the parameters for a Polynomial function to be used by 'call'
        The only thing that should be changed here is the parb values
        """
        pnumr = len(mp['mpar'][ival])
        levadd=0
        params=np.zeros(pnumr)
        parinf=[]
        for i in range(pnumr):
            lnkprm = None
            parb = dict({'ap_1a':self._keywd['scale']})
            if mp['mtie'][ival][i] >= 0:
                getid = mp['tpar'][mp['mtie'][ival][i]][1]
            elif mp['mtie'][ival][i] <= -2:
                if len(mp['mlnk']) == 0:
                    lnkprm = mp['mpar'][ival][i]
                else:
                    for j in range(len(mp['mlnk'])):
                        if mp['mlnk'][j][0] == mp['mtie'][ival][i]:
                            cmd = 'lnkprm = ' + mp['mlnk'][j][1]
                            exec(cmd)
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
            if nexbin[0] == "km/s": return params, 1
            elif nexbin[0] == "A" : return params, 1
            else: msgs.bug("bintype {0:s} should not have been specified in model function: {1:s}".format(nexbin[0],self._idstr), verbose=self._verbose)
        elif getinfl: return params, parinf
        else: return params

