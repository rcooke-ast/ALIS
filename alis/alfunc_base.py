from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np
import traceback
from alis import almsgs
import sys
msgs=almsgs.msgs()

class Base :
    """
    Returns a 1-dimensional gaussian of form:
    p[0] = amplitude
    p[1] = x offset
    p[2] = dispersion (sigma)
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'base'										# ID string for this class
        self._pnumr   = 3											# Total number of parameters fed in
        self._keywd   = dict({'specid':[], 'continuum':False, 'blind':False, 'wave':-1.0})			# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'specid':0,  'continuum':0,     'blind':0,     'wave':1})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'specid':"", 'continuum':"",    'blind':"",    'wave':""})			# Format for the keyword. "" is the Default setting
        self._parid   = ['amplitude', 'redshift', 'dispersion']		# Name of each parameter
        self._defpar  = [ 0.0,         0.0,        100.0 ]			# Default values for parameters that are not provided
        self._fixpar  = [ None,        None,       None ]			# By default, should these parameters be fixed?
        self._limited = [ [1  ,0  ],   [0  ,0  ], [1      ,0  ] ]	# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0],   [0.0,0.0], [1.0E-20,0.0] ]	# What should these limiting values be
        self._svfmt   = [ "{0:.8g}",   "{0:.8g}", "{0:.8g}"]		# Specify the format used to print or save output
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
        Define the functional form of the model for the CPU
        --------------------------------------------------------
        x  : array of wavelengths
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        def model():
            """
            Define the model here
            """
            return p[0]*np.exp(-(x-p[1])**2/(2.0*(p[2]**2)))
        #############
        yout = np.array((p.shape[0],x.size))
        for i in range(p.shape[0]):
            yout[i,:] = model()
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
        if self._fixpar[parj] not in ['False','false','FALSE']:
            adj = 1
            if self._fixpar[parj] not in ['True','true','TRUE']:
                if self._fixpar[parj] is True or self._fixpar[parj] is False: return mp # This means the default values were used and parameters are fixed
                try:
                    value = float(self._fixpar[parj])
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
            if self._limited[parj][jind] == 0: value = None
            else: value = float(self._limits[parj][jind])
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

    def getminmax(self, par, fitrng, Nsig=5.0):
        """
        This definition is only used for specifying the
        FWHM Resolution of the data.
        --------------------------------------------------------
        This definition will return the additional wavelength range
        of the data to be extracted around the user-speficied fitrange
        to ensure the edges of the model near the min and max of
        fitrange aren't affected.
        --------------------------------------------------------
        par    : The input parameters which defines the FWHM of this
                 function
        fitrng : The fitrange specified by the user at input
        Nsig   : Width in number of sigma to extract either side of
                 fitrange
        """
        return fitrng[0], fitrng[1]

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
            mps['mlim'][cntr].append([self._limits[iind][i] if self._limited[iind][i]==1 else None for i in range(2)])
            return mps
        ################
        isspl=instr.split()
        # Seperate the parameters from the keywords
        ptemp, kywrd = [], []
        keywdk = list(self._keywd.keys())
        keywdk[:] = (kych for kych in keywdk if kych[:] != 'input') # Remove the keyword 'input'
        numink = 0
        param = [None for all in range(self._pnumr)]
        parid = [i for i in range(self._pnumr)]
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                kwspl = isspl[i].split('=')
                if kwspl[0] in self._parid:
                    self._keywd['input'][kwspl[0]]=2
                    for j in range(self._pnumr):
                        if kwspl[0] == self._parid[j]:
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
        fixpid = [0 for all in range(self._pnumr)]
        for i in range(self._pnumr):
            if param[i] is None:
                if cntp >= len(ptemp): # Use default values
                    param[i] = str(self._defpar[i])
                    fixpid[i] = 1
                else:
                    param[i] = ptemp[cntp]
                    self._keywd['input'][self._parid[i]]=1
                    cntp += 1
        # Do some quick checks
        if len(ptemp)+numink > self._pnumr:
            msgs.error("Incorrect number of parameters (should be "+str(self._pnumr)+"):"+msgs.newline()+self._idstr+"   "+instr)
        if len(specid) > 1: # Force the user to specify a spectrum ID number
            self._keych['specid'] = 1
        specidset=False # Detect is the user has specified specid
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(self._pnumr):
            mp = check_tied_param(param[i], cntr, mp, i)
        if any(fixpid): # If default values had to be used, fix the values
            for i in range(self._pnumr):
                if fixpid[i] == 1: mp['mfix'][-1][i] = 1
        # Now load the keywords:
        # Make a copy of the default values
        cpy_keywd = self._keywd.copy()
        cpy_keych = self._keych.copy()
        for i in range(len(kywrd)):
            kwspl = kywrd[i].split('=')
            ksspl = kwspl[1].split(',')
            if kwspl[0] == 'specid':
                sidlist = []
                for j in range(len(ksspl)): # Do some checks
                    if ksspl[j] in sidlist:
                        msgs.error("specid: "+ksspl[j]+" is set twice for -"+msgs.newline()+self._idstr+"   "+instr.replace('\t','  '))
                    else: sidlist.append(ksspl[j])
                    if ksspl[j] not in specid:
                        msgs.error("There is no data with specid: "+ksspl[j]+" for -"+msgs.newline()+self._idstr+"   "+instr.replace('\t','  '))
                specidset=True
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
                elif type(cpy_keywd[kwspl[0]]) is bool:
                    if kwspl[1] in ['True', 'False']:
                        typeval='boolean'
                        cpy_keywd[kwspl[0]] = kwspl[1] in ['True']
                    else:
                        typeval='string'
                        cpy_keywd[kwspl[0]] = kwspl[1]
                        msgs.warn(kwspl[0]+" should be of type boolean (True/False)", verbose=self._verbose)
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
        if   i == 0: pin = par
        elif i == 1: pin = parb['ap_1a'] * (1.0 + par)
        elif i == 2: pin = parb['ap_2a'] * par/299792.458
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
        conv     : If convergence test is being written, conv is
                   the threshold for convergence (in sigma's).
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.
        --------------------------------------------------------
        """
        if errs is None: errors = params
        else: errors = errs
        add = self._pnumr
        havtie = 0
        tienum = 0
        levadd = 0
        outstring = ['  %s ' % (self._idstr)]
        errstring = ['# %s ' % (self._idstr)]
        for i in range(self._pnumr):
            if mp['mkey'][istart]['input'][self._parid[i]] == 0: # Parameter not given as input
                outstring.append( "" )
                errstring.append( "" )
                continue
            elif mp['mkey'][istart]['input'][self._parid[i]] == 1: pretxt = ""   # Parameter is given as input, without parid
            else: pretxt = self._parid[i]+"="                                    # Parameter is given as input, with parid
            if mp['mtie'][istart][i] >= 0:
                if reletter:
                    newfmt=pretxt+self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],self._svfmt[i]+'{1:c}')
                    outstring.append( (newfmt).format(params[mp['tpar'][mp['mtie'][istart][i]][1]],97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                    if conv is None:
                        errstring.append( (newfmt).format(errors[mp['tpar'][mp['mtie'][istart][i]][1]],97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                    else:
                        if params[mp['tpar'][mp['mtie'][istart][i]][1]] < conv: cvtxt = "CONVERGED"
                        else: cvtxt = "!!!!!!!!!"
                        errstring.append( ('--{0:s}--{1:c}    ').format(cvtxt,97+tienum-32*mp['mfix'][istart][1]) )
                else:
                    newfmt=pretxt+self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],self._svfmt[i]+'{1:s}')
                    outstring.append( (newfmt).format(params[mp['tpar'][mp['mtie'][istart][i]][1]],mp['tpar'][mp['mtie'][istart][i]][0]) )
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
                    if len(mp['tpar']) != 0:
                        if mp['tpar'][tienum][1] == level+levadd:
                            if reletter:
                                newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[i]+'{1:c}')
                                outstring.append( (newfmt).format(params[level+levadd],97+tienum-32*mp['mfix'][istart][1]) )
                                if conv is None:
                                    errstring.append( (newfmt).format(errors[level+levadd],97+tienum-32*mp['mfix'][istart][1]) )
                                else:
                                    if params[level+levadd] < conv: cvtxt = "CONVERGED"
                                    else: cvtxt = "!!!!!!!!!"
                                    errstring.append( ('--{0:s}--{1:c}    ').format(cvtxt,97+tienum-32*mp['mfix'][istart][1]) )
                            else:
                                newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[i]+'{1:s}')
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
                            newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[i])
                            outstring.append( (newfmt).format(params[level+levadd]) )
                            if conv is None:
                                errstring.append( (newfmt).format(errors[level+levadd]) )
                            else:
                                if params[level+levadd] < conv: cvtxt = "CONVERGED"
                                else: cvtxt = "!!!!!!!!!"
                                errstring.append( ('--{0:s}--    ').format(cvtxt) )
                    else: # There are no tied parameters!
                        newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[i])
                        outstring.append( (newfmt).format(params[level+levadd]) )
                        if conv is None:
                            errstring.append( (newfmt).format(errors[level+levadd]) )
                        else:
                            if params[level+levadd] < conv: cvtxt = "CONVERGED"
                            else: cvtxt = "!!!!!!!!!"
                            errstring.append( ('--{0:s}--    ').format(cvtxt) )
                else:
                    newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[i])
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
                        delind = self._pnumr+insind+j
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
                pinfo[level+levadd]['limited'] = [0 if o is None else 1 for o in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['limits']  = [0.0 if o is None else float(o) for o in mp['mlim'][mnum][i]]
                mp['mfix'][mnum][i] = -1
                levadd += 1
            else:
                pinfo[level+levadd]['limited'] = [0 if o is None else 1 for o in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['limits']  = [0.0 if o is None else float(o) for o in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['fixed']   = mp['mfix'][mnum][i]
                levadd += 1
        return pinfo, add

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, ddpid=None, getinfl=False):
        """
        Return the parameters for a Gaussian function to be used by 'call'
        The only thing that should be changed here is the parb values
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
            if nexbin[0] == "km/s": return params, 1
            elif nexbin[0] == "A" : return params, 1
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getinfl: return params, parinf
        else: return params

    def gtoef(self, num, fmt):
        """
        A definition to make the printout much nicer looking
        """
        return fmt.replace("g}","f}")+"    " if np.abs(np.log10(np.abs(num))) < 3. or num==0.0 else fmt.replace("g","E")

    def tick_info(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None):
        """
        For a given model, determine the wavelengths that
        tick marks should be plotted, and the label that
        should be associated with this tick mark.
        By default, return no tick marks and no labels.
        """
        return [], []
#######################################################################################
#######################################################################################
#######################################################################################

# Import all of the functions used in alis
from alis import alfunc_brokenpowerlaw
from alis import alfunc_chebyshev
from alis import alfunc_constant
from alis import alfunc_gaussian
from alis import alfunc_legendre
from alis import alfunc_linear
from alis import alfunc_lineemission
from alis import alfunc_phionxs
from alis import alfunc_polynomial
from alis import alfunc_powerlaw
from alis import alfunc_random
from alis import alfunc_spline
from alis import alfunc_splineabs
from alis import alfunc_thar
from alis import alfunc_tophat
from alis import alfunc_variable
from alis import alfunc_voigt
# Convolution+shifting routines
from alis import alfunc_afwhm
from alis import alfunc_apod
from alis import alfunc_lsf
from alis import alfunc_vfwhm
from alis import alfunc_voigtconv
from alis import alfunc_vsigma
from alis import alshift
from alis import alfunc_user

"""
Initialises the functions that are used in this file.
If you want your function to be recognised, you must
include it's idstr value and the function call here.
"""

def call(prgname="",getfuncs=False,getinst=False,atomic=None,verbose=2):
    sendatomic = ['voigt', 'lineemission', 'splineabs', 'phionxs']
    # Add your new function to the following:
    fd = dict({ 'Afwhm'          : alfunc_afwhm.AFWHM,
                'apod'           : alfunc_apod.APOD,
                'Ashift'         : alshift.Ashift,
                'brokenpowerlaw' : alfunc_brokenpowerlaw.BrokenPowerLaw,
                'chebyshev'      : alfunc_chebyshev.Chebyshev,
                'constant'       : alfunc_constant.Constant,
                'gaussian'       : alfunc_gaussian.Gaussian,
                'gaussianconst'  : alfunc_gaussian.GaussianConstant,
                'gaussianconsthelium'  : alfunc_gaussian.GaussianConstantHelium,
                'legendre'       : alfunc_legendre.Legendre,
                'linear'         : alfunc_linear.Linear,
                'lineemission'   : alfunc_lineemission.LineEmission,
                'phionxs'        : alfunc_phionxs.PhotIon_CrossSection,
                'polynomial'     : alfunc_polynomial.Polynomial,
                'polyshift'      : alshift.polyshift,
                'powerlaw'       : alfunc_powerlaw.PowerLaw,
                'random'         : alfunc_random.Random,
                'spline'         : alfunc_spline.Spline,
                'splineabs'      : alfunc_splineabs.SplineAbs,
                'thar'           : alfunc_thar.ThAr,
                'tophat'         : alfunc_tophat.TopHat,
                'variable'       : alfunc_variable.Variable,
                'vfwhm'          : alfunc_vfwhm.vFWHM,
                'lsf'            : alfunc_lsf.LSF,
                'voigt'          : alfunc_voigt.Voigt,
                'voigtconv'      : alfunc_voigtconv.VoigtConv,
                'vshift'         : alshift.vshift,
                'vshiftscale'    : alshift.vshiftscale,
                'vsigma'         : alfunc_vsigma.vSigma
                })

    # Load the user-specified functions
    msgs.info("Loading user functions")
    try:
        usr_fd, usr_atm = alfunc_user.load_user_functions()
    except Exception:
        msgs.warn("There appears to be a problem loading the user functions")
        et, ev, tb = sys.exc_info()
        while tb:
            co = tb.tb_frame.f_code
            filename = str(co.co_filename)
            line_no =  str(traceback.tb_lineno(tb))
            tb = tb.tb_next
        filename=filename.split('/')[-1]
        msgs.bug("A bug has been spotted on Line "+line_no+" of "+filename+" with error:"+msgs.newline()+str(ev))
        sys.exit()

    # Incorporate the user-defined functions
    kvals = list(usr_fd.keys())
    if len(kvals) == 0:
        msgs.info("No user functions to load!")
    fdk = list(fd.keys())
    # Check there is no overlap in function names
    for i in range(len(kvals)):
        if kvals[i] in fdk:
            msgs.error("There is already a built-in function called '{0:s}'".format(kvals[i])+msgs.newline()+"Please give your function a new name.")
        else:
            fd[kvals[i]] = usr_fd[kvals[i]]
            msgs.info("Successfully loaded user function: {0:s}".format(kvals[i]))
    for i in range(len(usr_atm)):
        if usr_atm[i] in kvals:
            sendatomic.append(usr_atm[i])
        else:
            msgs.error("with user-defined function {0:s}".format(usr_atm[i])+msgs.newline()+"Atomic data request for a function that does not exist!")

    # Don't touch anything below
    if getfuncs and getinst:
        msgs.bug("Two keywords in alfunc_base.py unexpectedly set to 'True' ...", verbose=2)
        sys.exit()
    if getinst:
        keys = list(fd.keys()) # Python 3
        for i in range(len(keys)):
            if keys[i] in sendatomic:
                fd[keys[i]] = fd[keys[i]](prgname=prgname, getinst=getinst, verbose=verbose, atomic=atomic)
            else:
                fd[keys[i]] = fd[keys[i]](prgname=prgname, getinst=getinst, verbose=verbose)
    if getfuncs:
        return list(fd.keys())
    else:
        return fd

