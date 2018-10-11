from __future__ import absolute_import, division, print_function

import numpy as np
import os, sys, imp
import copy
from alis import almsgs
from multiprocessing import cpu_count
msgs = almsgs.msgs()

from astropy.io import fits as pyfits

try: input = raw_input
except NameError: pass

try:
    from xastropy.xutils import xdebug as debugger
except:
    import pdb as debugger


def cpucheck(ncpu,curcpu=0,verbose=2):
    cpucnt=cpu_count()
    if ncpu == 'all':
        ncpu=cpucnt # Use all available cpus
        if cpucnt != curcpu: msgs.info("Setting %i CPUs" % (ncpu),verbose=verbose)
    elif ncpu == None:
        ncpu=cpucnt-1 # Use all but 1 available cpus
        if ncpu != curcpu: msgs.info("Setting %i CPUs" % (ncpu),verbose=verbose)
    else:
        try:
            ncpu = int(ncpu)
            if ncpu > cpucnt:
                msgs.warn("You don't have %i CPUs!" % (ncpu),verbose=verbose)
                ncpu = cpucnt
            elif ncpu < 0:
                ncpu += cpucnt
            if ncpu != curcpu: msgs.info("Setting %i CPUs" % (ncpu),verbose=verbose)
        except:
            msgs.error("Incorrect argument given for number of CPUs"+msgs.newline()+"Please choose from -"+msgs.newline()+"all, 1..."+str(cpucnt))
            if cpucnt == 1:
                if cpucnt != curcpu: msgs.info("Setting 1 CPU",verbose=verbose)
                ncpu=1
            else:
                ncpu=cpu_count()-1
                if ncpu != curcpu: msgs.info("Setting %i CPUs" % (ncpu),verbose=verbose)
    return ncpu


def usage(name):
    """ Header for parser
    Parameters
    ----------
    argflag

    Returns
    -------
    descs : str

    """
    #print "\n#####################################################################"
    descs =  msgs.alisheader(name, verbose=2)
    #descs += "\n##  -----------------------------------------------------------------"
    #descs += "\n##  Options: (default values in brackets)"
    #descs += "\n##   -c or --cpus      : (all) Number of cpu cores to use"
    #descs += "\n##   -f or --fits      : Write model fits to *.dat files"
    #descs += "\n##   -g or --gpu       : enable GPU multiprocessing"
    #descs += "\n##   -h or --help      : Print this message"
    #descs += "\n##   -j or --justplot  : Plot the data and input model - don't fit"
    #descs += "\n##   -l or --labels    : Label absorption components when plotting"
    #descs += "\n##   -m or --model     : Write an output model with fitting results"
    #descs += "\n##   -p or --plot      : ('2x3') plot model fits with MxN panels,"
    #descs += "\n##                             for no screen plots use: -p 0"
    #descs += "\n##   -r or random      : Number of random simulations to perform"
    #descs += "\n##   -s or --startid   : Starting ID number for the simulations"
    #descs += "\n##   -v or --verbose   : (2) Level of verbosity (0-2)"
    #descs += "\n##   -w or --writeover : If output files exist, write over them"
    #descs += "\n##   -x or --xaxis     : (0) Plot observed/rest wavelength [0/1]"
    #descs += "\n##                           or velocity [2]"
    #descs += "\n##  -----------------------------------------------------------------"
    #descs += "\n##  {0:s}".format(argflag['run']['last_update'])
    #descs += "\n#####################################################################\n"
    #sys.exit()
    # Return
    return descs


def optarg(pathname, argv=None, verbose=2):

    # Load the default settings
    prgn_spl = pathname.split('/')
    try:
        fname = ""
        for i in range(0, len(prgn_spl)-2):
            fname += prgn_spl[i]+"/"
        fname += 'data/settings.alis'
        argflag = load_settings(fname, verbose=verbose)
    except IOError:
        fname = ""
        for i in range(0, len(prgn_spl)-1):
            fname += prgn_spl[i]+"/"
        fname += 'data/settings.alis'
        argflag = load_settings(fname, verbose=verbose)
    argflag['run']['prognm'] = __file__  #pathname
    """
    if argv is not None:
        # Load options from command line
        try:
            opt,arg=getopt.getopt(argv,'hc:p:x:v:r:s:gfmjwxl', ['help',
                                                         'cpus',
                                                         'plot',
                                                         'xaxis',
                                                         'verbose',
                                                         'random',
                                                         'startid',
                                                         'gpu',
                                                         'fits',
                                                         'model',
                                                         'justplot',
                                                         'writeover',
                                                         'labels',
                                                        ])
        except getopt.GetoptError, err:
            msgs.error(err.msg)
            usage(argflag)
    """
    plxaxis = ['observed','rest','velocity']
    if argv is not None:
        if argv.cpus is not None:
            argflag['run']['ncpus']   = argv.cpus
        if argv.gpu:
            argflag['run']['ngpus']   = argv.gpu
        if argv.plot is not None:
            argflag['plot']['dims']   = argv.plot
        if argv.xaxis is not None:
            argflag['plot']['xaxis']  = plxaxis[argv.xaxis]
        if argv.justplot:
            argflag['plot']['only']   = argv.justplot
        if argv.labels:
            argflag['plot']['labels']  = argv.labels
        if argv.verbose is not None:
            argflag['out']['verbose']  = argv.verbose
        if argv.random is not None:
            argflag['sim']['random']   = argv.random
        if argv.startid is not None:
            argflag['sim']['startid']  = argv.startid
        if argv.fits:
            argflag['out']['fits']     = argv.fits
        if argv.model:
            argflag['out']['model']    = argv.model
        if argv.outname:
            argflag['out']['modelname']    = argv.outname
        if argv.writeover:
            argflag['out']['overwrite'] = argv.writeover
    #######################
    # Now do some checks: #
    #######################

    # Check requested CPUs
    argflag['run']['ncpus'] = cpucheck(argflag['run']['ncpus'],verbose=verbose)

    # Check that fits files are being generated if supermongo scripts are generated
    if argflag['out']['sm'] == True and argflag['out']['fits'] == False:
        msgs.warn("You must set the 'fits' flag if you want to"+msgs.newline()+"produce a SuperMongo file",verbose=verbose)
        msgs.info("Setting the fits flag",verbose=verbose)
        argflag['out']['fits'] = True

    return argflag

def set_params(lines, argflag, setstr=""):
    """
    Adjust settings parameters.
    lines    : an array of settings with the same format as the default 'settings.alis'
    argflag  : a dictionary generated by initialise that contains all settings
    setstr   : a string argument for error messages that tells the user which file the error occured in.
    """
    for i in range(len(lines)):
        if lines[i] == '': continue
        if lines[i][0] == '#' or lines[i] == '\n': continue
        linspl = lines[i].split()
        if linspl[0] in list(argflag.keys()):
            if linspl[1] in argflag[linspl[0]].keys():
                try:
                    if type(argflag[linspl[0]][linspl[1]]) is int:
                        argflag[linspl[0]][linspl[1]] = int(linspl[2])
                    elif type(argflag[linspl[0]][linspl[1]]) is str:
                        argflag[linspl[0]][linspl[1]] = linspl[2]
                    elif type(argflag[linspl[0]][linspl[1]]) is float:
                        argflag[linspl[0]][linspl[1]] = float(linspl[2])
                    elif type(argflag[linspl[0]][linspl[1]]) is list:
                        argflag[linspl[0]][linspl[1]].append(linspl[2])
                    elif type(argflag[linspl[0]][linspl[1]]) is bool:
                        argflag[linspl[0]][linspl[1]] = linspl[2] in ['True']
                    elif argflag[linspl[0]][linspl[1]] is None:
                        if linspl[2].lower() == 'none': # None type
                            argflag[linspl[0]][linspl[1]] = None
                        elif linspl[2].lower() == 'true' or linspl[2].lower() == 'false': # bool type
                            argflag[linspl[0]][linspl[1]] = linspl[2].lower() in ['true']
                        elif ',' in linspl[2]: # a list
                            argflag[linspl[0]][linspl[1]] = linspl[2].split(',')
                        elif '.' in linspl[2]: # Might be a float
                            try: argflag[linspl[0]][linspl[1]] = float(linspl[2])
                            except: argflag[linspl[0]][linspl[1]] = linspl[2] # Must be a string
                        else:
                            try: argflag[linspl[0]][linspl[1]] = int(linspl[2]) # Could be an integer
                            except: argflag[linspl[0]][linspl[1]] = linspl[2] # Must be a string
                    else:
                        msgs.error(setstr + "Settings contains bad line (arg 3):"+msgs.newline()+lines[i].split('#')[0].strip())
                except:
                    msgs.error(setstr + "Settings contains bad line (arg 3):"+msgs.newline()+lines[i].split('#')[0].strip())
            else: msgs.error(setstr + "Settings contains bad line (arg 2):"+msgs.newline()+lines[i].split('#')[0].strip())
        else: msgs.error(setstr + "Settings contains bad line (arg 1):"+msgs.newline()+lines[i].split('#')[0].strip())
    return argflag


def load_settings(fname,verbose=2):
    def initialise():
        """
        Initialise the default settings called argflag
        """
        rna = dict({'prognm':'alis.py', 'last_update':'Last updated 25th March 2015', 'atomic':'atomic.xml', 'modname':'model.mod', 'convergence':False, 'convnostop':False, 'convcriteria':0.2, 'datatype':'default', 'limpar':False, 'ncpus':-1, 'ngpus':None, 'nsubpix':5, 'nsubmin':5, 'nsubmax':21, 'warn_subpix':100, 'renew_subpix':False, 'blind':True, 'bintype':'km/s', 'logn':True, 'capvalue':None})
        csa = dict({'miniter':0, 'maxiter':20000, 'atol':1.0E-10, 'xtol':1.0E-10, 'ftol':1.0E-10, 'gtol':1.0E-10, 'fstep':1.0})
        pla = dict({'dims':'3x3', 'fits':True, 'residuals':False, 'xaxis':'observed', 'labels':False, 'only':False, 'pages':'all', 'ticks':True, 'ticklabels':False, 'fitregions':False})
        opa = dict({'model':True, 'modelname':'','plots':'', 'fits':False, 'onefits':False, 'overwrite':False, 'sm':False, 'verbose':2, 'reletter':False, 'covar':"", 'convtest':""})
        mca = dict({'random':None, 'perturb':None, 'systematics':False, 'beginfrom':"", 'startid':0, 'maxperturb':0.1, 'systmodule':None, 'newstart':True, 'dirname':'sims', 'edgecut':4.0})
        gna = dict({'pixelsize':2.5, 'data':False, 'overwrite':False, 'peaksnr':0.0, 'skyfrac':0.0})
        itr = dict({'model':None, 'data':None})
        argflag = dict({'run':rna, 'chisq':csa, 'plot':pla, 'out':opa, 'sim':mca, 'generate':gna, 'iterate':itr})
        return argflag

    # Read in the default settings
    msgs.info("Loading the default settings", verbose=verbose)
    argflag = initialise()
    infile = open(fname, 'r')
    lines = infile.readlines()
    argflag = set_params(lines, argflag, setstr="Default ")
    return argflag

def check_argflag(argflag, curcpu=None):
    # Make some final corrections to the input parameters
    argflag['out']['covar'] = argflag['out']['covar'].strip("\"'")
    argflag['out']['plots'] = argflag['out']['plots'].strip("\"'")
    argflag['out']['convtest'] = argflag['out']['convtest'].strip("\"'")
    argflag['sim']['beginfrom'] = argflag['sim']['beginfrom'].strip("\"'")
    # Check requested CPUs
    argflag['run']['ncpus'] = cpucheck(argflag['run']['ncpus'], curcpu=curcpu, verbose=argflag['out']['verbose'])
    # Change some arguments if the analysis is to be blind
    if argflag['run']['blind']:
        msgs.info("Running a blind analysis",verbose=argflag['out']['verbose'])
        if argflag['plot']['dims'] == '0':
            msgs.warn("For a blind analysis, you may want to plot the fits.",verbose=argflag['out']['verbose'])
        if argflag['out']['verbose'] == 2:
            msgs.warn("For a blind analysis, verbosity must be minimal.",verbose=argflag['out']['verbose'])
            argflag['out']['verbose'] = 1
        if argflag['out']['model']:
            msgs.warn("For a blind analysis, you cannot output the best-fitting model.",verbose=argflag['out']['verbose'])
            argflag['out']['model'] = False
        if argflag['out']['fits'] or argflag['out']['onefits']:
            msgs.warn("For a blind analysis, you cannot save the best-fitting model profiles.",verbose=argflag['out']['verbose'])
            argflag['out']['fits'] = False
            argflag['out']['onefits'] = False
        if argflag['out']['sm'] or argflag['out']['sm']:
            msgs.warn("For a blind analysis, you cannot save a Supermongo file.",verbose=argflag['out']['verbose'])
            argflag['out']['sm'] = False
        if argflag['out']['covar'] != "":
            msgs.warn("For a blind analysis, you cannot save the covariance matrix.",verbose=argflag['out']['verbose'])
            argflag['out']['covar'] = ""
    # Perform some checks on the input parameters:
    if argflag['plot']['only'] == True:
        msgs.warn("When plot+only is True, you must set plot+fits to be True",verbose=argflag['out']['verbose'])
        argflag['plot']['fits'] = True
    if argflag['chisq']['fstep'] < 1.0: msgs.error("Setting 'fstep' in family 'run' must be >= 1.0")
    if argflag['out']['verbose'] not in [-1,0,1,2]: msgs.error("Setting 'verbose' in family 'out' must equal -1, 0, 1, or 2.")
    if argflag['chisq']['miniter'] > argflag['chisq']['maxiter']:
        msgs.warn("Setting 'miniter' is greater than 'maxiter' in family 'chisq'",verbose=argflag['out']['verbose'])
        msgs.info("Forcing maximum number of iterations to be {0:d}".format(argflag['chisq']['maxiter']),verbose=argflag['out']['verbose'])
    # Make some checks on the convergence parameters
    if argflag['run']['convnostop'] and not argflag['run']['convergence']:
        msgs.warn("Setting `convnostop' is True, but setting `convergence' is False."+msgs.newline()+"Not performing a convergence check",verbose=argflag['out']['verbose'])
    if argflag['run']['convcriteria'] >= 1.0:
        msgs.warn("Convergence criteria is greater than 1 standard deviation."+msgs.newline()+"Are you sure you want to do this?",verbose=argflag['out']['verbose'])
    # Check maxperturb is positive
    if argflag['sim']['maxperturb'] < 0.0: argflag['sim']['maxperturb'] *= -1.0
    # If the user specified their own systematics module, try to import
    # it here to make sure it exists and there are no mistakes.
    if type(argflag['sim']['systmodule']) is str: argflag['sim']['systmodule'] = argflag['sim']['systmodule'].replace('.py','')
    builtinsyst = [None, 'default', 'continuumpoly']
    if argflag['sim']['random'] is not None and argflag['sim']['systematics'] and argflag['sim']['systmodule'] not in builtinsyst:
        try:
            usrmod = __import__(argflag['sim']['systmodule'])
            if 'loader' not in dir(usrmod): msgs.error("Systematics module {0:s} must contain a function called 'loader'".format(argflag['sim']['systmodule']))
        except: msgs.error("Could not import module {0:s}".format(argflag['sim']['systmodule']))
    return

def load_input(slf, filename=None, textstr=None, updateself=True):
    # Read in the model file
    if textstr is not None:
        if type(textstr) is str:
            lines = textstr.split("\n")
        elif type(textstr) is list:
            lines = textstr
        else:
            msgs.error("The keyword value for textstr in load.load_input is not understood")
    else:
        msgs.info("Loading the input file",verbose=slf._argflag['out']['verbose'])
        if filename is None:
            loadname = slf._argflag['run']['modname']
        else:
            loadname = filename
        if loadname.split(".")[-1] == "fits":
            # This could be an ALIS onefits file.
            parlines, datlines, modlines, lnklines = load_onefits(slf,loadname)
            argflag = set_params(parlines, copy.deepcopy(slf._argflag), setstr="OneFits ")
            # Check the settings are OK.
            check_argflag(argflag, curcpu=slf._argflag['run']['ncpus'])
            msgs.info("Input file loaded successfully",verbose=slf._argflag['out']['verbose'])
            if updateself:
                slf._argflag = argflag
            return parlines, datlines, modlines, lnklines
        try:
            infile = open(loadname, 'r')
        except IOError:
            msgs.error("The filename does not exist -"+msgs.newline()+loadname)
            sys.exit()
        lines = infile.readlines()
    # Separate the lines into par-, dat-, and mod- lines.
    parlines = []
    datlines = []
    modlines = []
    lnklines = []
    rddata, rdmodl, rdlink = 0, 0, 0
    ignlin = -1
    for i in range(len(lines)):
        goodtxt = lines[i].strip().split('#')[0]
        # Check if we are currently ignoring sections of the input script
        if ignlin != -1:
            if '<--#' in lines[i]:
                goodtxt = lines[i].strip().split('<--#')[1]
                if '#-->' in goodtxt:
                    msgs.error("You can only use one long comment (either '#-->' or '<--#') per line")
                msgs.warn("Ignoring lines {0:d} --> {1:d} of input file".format(ignlin,i+1),verbose=slf._argflag['out']['verbose'])
                ignlin = -1
            else: continue
            if len(goodtxt) == 0: continue
        elif '<--#' in lines[i]:
            msgs.error("I encountered the end of a long comment ('<--#') on line {0:d} of".format(i+1)+msgs.newline()+"the input file, before the start of a long comment ('#-->')")
        # Check if we need to start ignoring chunks of input
        if '#-->' in lines[i]:
            ignlin = i+1
            templin = lines[i].split('#-->')
            if '<--#' in templin[1]:
                msgs.error("You can only use one long comment (either '#-->' or '<--#') per line")
        # Check if there's nothing on a line
        if goodtxt == '': continue
        linspl = goodtxt.split()
        if rddata == 1:
            if linspl[0] == 'data' and linspl[1] == 'end':
                rddata += 1
                continue
            datlines.append(lines[i])
            continue
        elif rddata == 0 and linspl[0] == 'data' and linspl[1] == 'read':
            rddata += 1
            continue
        if rdmodl == 1:
            if linspl[0] == 'model' and linspl[1] == 'end':
                rdmodl += 1
                continue
            modlines.append(lines[i])
            continue
        elif rdmodl == 0 and linspl[0] == 'model' and linspl[1] == 'read':
            rdmodl += 1
            continue
        if rdlink == 1:
            if linspl[0] == 'link' and linspl[1] == 'end':
                rdlink += 1
                continue
            lnklines.append(lines[i])
            continue
        elif rdlink == 0 and linspl[0] == 'link' and linspl[1] == 'read':
            rdlink += 1
            continue
        if lines[i].lstrip()[0] == '#': continue
        parlines.append(lines[i])
    # Do some quick checks
    if rddata == 0:
        msgs.error("You haven't specified any data!")
    elif rddata != 2:
        msgs.error("Missing 'data end' in "+loadname)
    if rdmodl == 0:
        msgs.error("You haven't specified a model!")
    elif rddata != 2:
        msgs.error("Missing 'model end' in "+loadname)
    # Now update the settings
    argflag = set_params(parlines, copy.deepcopy(slf._argflag), setstr="Model ")
    # Check the settings are OK.
    check_argflag(argflag, curcpu=slf._argflag['run']['ncpus'])
    msgs.info("Input file loaded successfully",verbose=slf._argflag['out']['verbose'])
    if updateself:
        slf._argflag = argflag
    return parlines, datlines, modlines, lnklines

def load_atomic(slf):
    """
    Load the atomic transitions data
    """
    from astropy.io.votable import parse_single_table

    prgname, atmname = slf._argflag['run']['prognm'], slf._argflag['run']['atomic']
    msgs.info("Loading atomic data", verbose=slf._argflag['out']['verbose'])
    prgn_spl = prgname.split('/')
    fname = ""
    for i in range(0,len(prgn_spl)-1): fname += prgn_spl[i]+"/"
    fname += "data/"
    # If the user specifies the atomic data file, make sure that it exists
    if os.path.exists(fname+atmname):
        msgs.info("Using atomic datafile:"+msgs.newline()+fname+atmname, verbose=slf._argflag['out']['verbose'])
    else:
        msgs.warn("Couldn't find atomic datafile:"+msgs.newline()+fname+atmname, verbose=slf._argflag['out']['verbose'])
        msgs.info("Using atomic datafile:"+msgs.newline()+fname+"atomic.xml", verbose=slf._argflag['out']['verbose'])
        atmname="atomic.xml"
    fname += atmname
    try:
        table = parse_single_table(fname)
    except IOError:
        msgs.error("The filename does not exist -"+msgs.newline()+fname)
    isotope = table.array['MassNumber'].astype("|S3").astype(np.object)+table.array['Element']
    atmdata = dict({})
    # eln = Ion
    # elw = Wavelength
    # elt = Gamma
    # elf = fvalue
    # elq = Qvalue
    # elK = Kvalue
    # elname = Element
    # elmass = AtomicMass
    atmdata['Ion'] = np.array(isotope+b"_"+table.array['Ion']).astype(np.str)
    atmdata['Wavelength'] = np.array(table.array['RestWave'])
    atmdata['fvalue'] = np.array(table.array['fval'])
    atmdata['Gamma'] = np.array(table.array['Gamma'])
    atmdata['Qvalue'] = np.array(table.array['q'])
    atmdata['Kvalue'] = np.array(table.array['K'])
    seen = set()
    atmdata['Element'] = np.array([x for x in isotope if x not in seen and not seen.add(x)]).astype(np.str)
    seen = set()
    atmdata['AtomicMass'] = np.array([x for x in table.array['AtomicMass'] if x not in seen and not seen.add(x)])
    return atmdata

def load_data(slf, datlines, data=None):
    """
    Load the observed data
    """
    msgs.info("Loading the data",verbose=slf._argflag['out']['verbose'])
    # Do some checks
    if data is not None and len(datlines) > 1:
        msgs.error("If you have used the data keyword to pass an array of data,"+msgs.newline()+"the keyword datlines can have a maximum of one element")
    if data is not None:
        datshp = data.shape
        if datshp[0] < 3:
            msgs.error("When specifying the data with keyword 'data', you must provide"+msgs.newline()+
                        "at least 3 rows of data (including, wave, flux, and error information)"+msgs.newline()+
                        "use a numpy array with shape (3,*) where * represents the number of data points")
    # Now begin
    specid=np.array([])
    slf._snipid=[]
    datopt = dict({'specid':[],'fitrange':[],'loadrange':[],'plotone':[],'nsubpix':[],'bintype':[],'columns':[],'systematics':[],'systmodule':[],'label':[],'yrange':[]})
    keywords = ['specid','fitrange','loadrange','systematics','systmodule','resolution','shift','columns','plotone','nsubpix','bintype','loadall','label','yrange']
    colallow = np.array(['wave','flux','error','continuum','zerolevel','fitrange','loadrange','systematics','resolution'])
    columns  = np.array(['wave','flux','error'])
    systload = [None, 'continuumpoly']
    # Get all SpecIDs and check if input is correct
    for i in range(0,len(datlines)):
        if len(datlines[i].strip()) == 0 : continue # Nothing on a line
        nocoms = datlines[i].lstrip().split('#')[0] # Remove everything on a line after the first instance of a comment symbol: #
        if len(nocoms) == 0: continue # A comment line
        linspl = nocoms.split()
        if data is None:
            filename = linspl[0]
        else:
            filename = "Data read from 'data' keyword"
            linspl.insert(0,linspl[0])
        if not os.path.exists(filename) and data is None:
            if not slf._argflag['generate']['data'] and not slf._isonefits:
                msgs.error("File does not exist -"+msgs.newline()+filename)
        fitfromcol, loadfromcol, systfromcol, resnfromcol, specidgiven = False, False, False, False, False
        fitspl, loadspl, systfrom, resnfrom = [''], [''], '', ''
        if len(linspl) > 1:
            for j in range(1,len(linspl)):
                if '=' not in linspl[j]: msgs.error("read data has bad form on data line: "+str(i+1))
                kwdspl = linspl[j].split('=')
                if   kwdspl[0] not in keywords: msgs.error("read data has bad keyword "+kwdspl[0]+" on data line: "+str(i+1))
                if   kwdspl[0] == 'specid':
                    if kwdspl[1] not in specid: specid = np.append(specid, kwdspl[1])
                    specidgiven = True
                    slf._snipid.append(kwdspl[1])
                elif kwdspl[0] == 'columns':
                    colsplt = kwdspl[1].strip('[]()').split(',')
                    colspl = [cs.split(":")[0].strip() for cs in colsplt]
                    if 'flux' not in colspl or 'error' not in colspl:
                        msgs.error("read data 'columns' must at least have keywords: "+msgs.newline()+"flux, error"+msgs.newline()+"and also the keyword 'wave' if input is ascii-formatted.")
                    for k in range(len(colspl)):
                        if colspl[k].strip() not in colallow: msgs.error("read data 'columns' cannot have value: "+colspl[k])
                        if colspl[k].strip() == 'fitrange': fitfromcol = True
                        elif colspl[k].strip() == 'loadrange': loadfromcol = True
                        elif colspl[k].strip() == 'systematics': systfromcol = True
                        elif colspl[k].strip() == 'resolution': resnfromcol = True
                elif kwdspl[0] == 'fitrange':
                    fitspl = kwdspl[1].strip('[]()').split(',')
                    if fitspl != ['all'] and fitspl != ['columns'] and np.size(np.shape(fitspl)) != 1:
                        msgs.error("read data 'fitrange' cannot have the value: "+colspl[k])
                    if len(fitspl) == 2:
                        try:
                            wavemin, wavemax = float(fitspl[0]), float(fitspl[1])
                        except:
                            msgs.error("read data has bad 'fitrange' wavelength range: "+msgs.newline()+linspl[0])
                        if wavemax <= wavemin: msgs.error("Upper limit on fitrange wavelength should be > lower limit for:"+msgs.newline()+datlines[i])
                    elif len(fitspl) == 1: pass
                    else: msgs.error("read data has bad 'fitrange' wavelength range: "+msgs.newline()+linspl[0])
                elif kwdspl[0] == 'loadrange':
                    loadspl = kwdspl[1].strip('[]()').split(',')
                    if loadspl != ['all'] and np.size(np.shape(loadspl)) != 1:
                        msgs.error("read data 'loadrange' cannot have the value: "+colspl[k])
                    if len(loadspl) == 2:
                        try:
                            lwavemin, lwavemax = float(loadspl[0]), float(loadspl[1])
                        except:
                            msgs.error("read data has bad 'loadrange' wavelength range: "+msgs.newline()+linspl[0])
                        if lwavemax <= lwavemin: msgs.error("Upper limit on loadrange wavelength should be > lower limit for:"+msgs.newline()+datlines[i])
                    elif len(loadspl) == 1: pass
                    else: msgs.error("read data has bad 'loadrange' wavelength range: "+msgs.newline()+linspl[0])
                elif kwdspl[0] == 'systematics':
                    systfrom=kwdspl[1]
                    if kwdspl[1] != 'columns' and kwdspl[1].lower() != 'none':
                        if not os.path.exists(kwdspl[1]):
                            msgs.warn("File doesn't exist:"+msgs.newline()+kwdspl[1],verbose=slf._argflag['out']['verbose'])
                            msgs.error("read data 'systematics' cannot have the value: {0:s}.".format(kwdspl[1]))
                elif kwdspl[0] == 'resolution':
                    resnfrom=kwdspl[1]
                elif kwdspl[0] == 'systmodule':
                    if kwdspl[1].lower() != 'none':
                        if ',' in kwdspl[1]: module, defn = kwdspl[1].split(',')
                        else: module, defn = kwdspl[1], ''
                        if '.py' not in module:
                            msgs.warn("You should include the extension '.py' for the systematics module.",verbose=slf._argflag['out']['verbose'])
                            module += '.py'
                        if kwdspl[1] not in systload:
                            msgs.info("Attempting to import module: {0:s}".format(module),verbose=slf._argflag['out']['verbose'])
                            try: usrmod = imp.load_source('loader',module)
                            except: msgs.error("Could not import module {0:s}".format(module))
                            if 'loader' not in dir(usrmod): msgs.error("Systematics module {0:s} must contain function 'loader'".format(module))
#							if defn != '':
#								if defn not in dir(usrmod): msgs.error("Systematics module {0:s} must contain function {1:s}".format(module,defn))
                            systload.append(kwdspl[1])
                elif kwdspl[0] == 'bintype':
                    if kwdspl[1] not in ['km/s','A','Hz']: msgs.error("Bintype "+kwdspl[1]+" is not allowed")
            if fitspl == ['columns'] and not fitfromcol:
                msgs.error("You must specify which column fitrange is in")
            if loadspl == ['columns'] and not loadfromcol:
                msgs.error("You must specify which column loadrange is in")
            if systfrom == 'columns' and not systfromcol:
                msgs.error("You must specify which column systematics information is in")
            if resnfrom == 'columns' and not resnfromcol:
                msgs.error("You must specify which column resolution information is in")
        if not specidgiven:
            if 'None' not in specid: specid = np.append(specid, 'None')
            slf._snipid.append('None')
    # Prepare the data arrays
    snipnames, resn, shft, posnfull, posnfit, plotone = [], [], [], [], [], []
    wavefull, fluxfull, fluefull, contfull, zerofull, systfull = [], [], [], [], [], []
    wavefit, fluxfit, fluefit, contfit, zerofit = [], [], [], [], []
    for i in range(specid.size):
        datopt['plotone'].append([])
        datopt['columns'].append([])
        datopt['systematics'].append([])
        datopt['systmodule'].append([])
        datopt['nsubpix'].append([])
        datopt['bintype'].append([])
        datopt['label'].append([])
        datopt['yrange'].append([])
        datopt['specid'].append([])
        datopt['fitrange'].append([])
        datopt['loadrange'].append([])
        snipnames.append([])
        posnfull.append([])
        wavefull.append( np.array([]) )
        fluxfull.append( np.array([]) )
        fluefull.append( np.array([]) )
        contfull.append( np.array([]) )
        zerofull.append( np.array([]) )
        systfull.append( np.array([]) )
        resn.append( np.array([]) )
        shft.append( np.array([]) )
        posnfit.append([])
        wavefit.append( np.array([]) )
        fluxfit.append( np.array([]) )
        fluefit.append( np.array([]) )
        contfit.append( np.array([]) )
        zerofit.append( np.array([]) )
    # Load data for each specid.
    if len(specid) == 1: sind = 0
    datnum=1
    for i in range(0,len(datlines)):
        if len(datlines[i].strip()) == 0 : continue # Nothing on a line
        nocoms = datlines[i].lstrip().split('#')[0] # Remove everything on a line after the first instance of a comment symbol: #
        if len(nocoms) == 0: continue # A comment line
        wfe = dict({'wave':0, 'flux':1, 'error':2, 'continuum':-1, 'zerolevel':-1, 'systematics':-1, 'fitrange':-1, 'loadrange':-1, 'resolution':-1})
        fitrange = 'all'
        linspl = nocoms.split()
        if data is None:
            filename = linspl[0]
        else:
            filename = "Data read from 'data' keyword"
            linspl.insert(0,linspl[0])
        setplot,loadall=False,False
        uselabel  = ''
        specidtxt = ''
        fitrtxt   = ''
        loadrtxt   = ''
        yrngtxt   = ''
        wavemin, wavemax = None, None
        lwavemin, lwavemax = None, None
        nspix = slf._argflag['run']['nsubpix']
        bntyp = slf._argflag['run']['bintype']
        tempresn = 'vfwhm(0.0)' # If Resolution not specified set it to zero and make it not tied to anything
        tempshft = 'none' # By default, apply no shift
        systfile = ''
        systval = None
        if len(linspl) > 1:
            for j in range(1,len(linspl)):
                kwdspl = linspl[j].split('=')
                if   kwdspl[0] == 'specid':
                    sind = np.where(specid == kwdspl[1])[0][0]
                    specidtxt = linspl[j]
                elif kwdspl[0] == 'resolution':
                    if kwdspl[1].lower() == 'none':
                        tempresn = 'vfwhm(0.0)'
                    elif kwdspl[1] == 'columns':
                        tempresn = kwdspl[1]
                    else:
                        tempresn = kwdspl[1]
                elif kwdspl[0] == 'shift':
                    if kwdspl[1].lower() == 'none':
                        tempshft = 'none'
                    else:
                        tempshft = kwdspl[1]
                elif kwdspl[0] == 'nsubpix':
                    if int(kwdspl[1]) > 0: nspix = int(kwdspl[1])
                    else: msgs.error("nsubpix must be greater than 0")
                elif kwdspl[0] == 'bintype': bntyp = kwdspl[1]
                elif kwdspl[0] == 'fitrange':
                    fitrtxt = linspl[j]
                    fitspl = kwdspl[1].strip('[]()').split(',')
                    if len(fitspl) == 2:
                        wavemin, wavemax = np.float64(fitspl[0]), np.float64(fitspl[1])
                    elif fitspl == ['all']:
                        wavemin, wavemax = -1.0, -1.0
                    elif fitspl == ['columns']:
                        wavemin, wavemax = -2.0, -2.0
                    else: msgs.bug("specified 'fitrange' is not allowed on line -"+msgs.newline()+datlines[i],verbose=slf._argflag['out']['verbose'])
                elif kwdspl[0] == 'loadrange':
                    loadrtxt = linspl[j]
                    loadspl = kwdspl[1].strip('[]()').split(',')
                    if len(loadspl) == 2:
                        lwavemin, lwavemax = np.float64(loadspl[0]), np.float64(loadspl[1])
                    elif loadspl == ['all']:
                        lwavemin, lwavemax = -1.0, -1.0
                    elif loadspl == ['columns']:
                        lwavemin, lwavemax = -2.0, -2.0
                    else: msgs.bug("specified 'loadrange' is not allowed on line -"+msgs.newline()+datlines[i],verbose=slf._argflag['out']['verbose'])
                elif kwdspl[0] == 'systematics':
                    if kwdspl[1] == 'columns': pass
                    else: systfile = kwdspl[1]
                elif kwdspl[0] == 'systmodule':
                    if kwdspl[1].lower() == 'none': systval='none'
                    else:
                        systval = kwdspl[1]
                        if '.py' not in kwdspl[1]: systval = systval.replace(',','.py,')
                elif kwdspl[0] == 'columns':
                    colspl = kwdspl[1].strip('[]()').split(',')
                    for k in range(len(colspl)):
                        clspid = colspl[k].strip().split(':')
                        if len(clspid)==2: wfe[clspid[0].strip()] = np.int(clspid[1].strip())
                        else: wfe[clspid[0].strip()] = k
                elif kwdspl[0] == 'plotone' and kwdspl[1].lower() == 'true':
                    setplot = True # Force this plot to be shown by itself
                elif kwdspl[0] == 'label':
                    uselabel = kwdspl[1]
                elif kwdspl[0] == 'yrange':
                    yrngtxt = linspl[j]
                    yrngspl = kwdspl[1].strip('[]()').split(',')
                    if len(yrngspl) == 2:
                        if yrngspl[0].lower() == "none":
                            yrngmin = None
                        else:
                            yrngmin = np.float64(yrngspl[0])
                        if yrngspl[1].lower() == "none":
                            yrngmax = None
                        else:
                            yrngmax = np.float64(yrngspl[1])
                    elif yrngspl == ['all']:
                        yrngmin, yrngmax = None, None
                    else: msgs.bug("specified 'yrange' is not allowed on line -"+msgs.newline()+datlines[i],verbose=slf._argflag['out']['verbose'])
                elif kwdspl[0] == 'loadall' and kwdspl[1] in ['True','true','TRUE']:
                    msgs.warn("The command 'loadall' is deprecated"+msgs.newline()+"Please use loadrange=all")
                    msgs.info("Loading the entire wavelength range")
                    lwavemin, lwavemax = -1.0, -1.0
        if wavemin is None or wavemax is None:
            msgs.error("A fitted wavelength range has not been specified")
#		if True:
        try:
            # Check first if we are supposed to generate our own data
            if slf._argflag['generate']['data']:
                if os.path.exists(filename) and not slf._argflag['generate']['overwrite']:
                    msgs.info("Reading in the following file to generate data:"+msgs.newline()+filename,verbose=slf._argflag['out']['verbose'])
                    wavein, fluxin, fluein, contin, zeroin, systin, fitrin, loadin = load_datafile(filename, colspl, wfe, verbose=slf._argflag['out']['verbose'])
                else:
                    if os.path.exists(filename) and slf._argflag['generate']['overwrite']:
                        msgs.warn("Overwriting the following file:"+msgs.newline()+filename,verbose=slf._argflag['out']['verbose'])
                        os.remove(filename)
                    else:
                        msgs.warn("The following file does not exist:"+msgs.newline()+filename,verbose=slf._argflag['out']['verbose'])
                    msgs.info("Generating a wavelength array for this file",verbose=slf._argflag['out']['verbose'])
                    # Do some checks
                    if lwavemin is None or lwavemax is None:
                        if wavemin is None or wavemax is None:
                            msgs.error("When generating data, use the loadrange command to specify"+msgs.newline()+"the wavelength range of the generated data")
                        else:
                            lwavemin = wavemin
                            lwavemax = wavemax
                    if lwavemin < 0.0 or lwavemax < 0.0:
                        msgs.error("Please check the minimum and maximum wavelength ranges")
                    # Generate the data
#					if slf._argflag['generate']['peaksnr'] <= 0.0: nsr = 0.0 # Noise-to-signal ratio
#					else: nsr = 1.0/slf._argflag['generate']['peaksnr']
                    if bntyp == 'km/s':
                        npix = 1.0+np.ceil(np.log10(lwavemax/lwavemin)/np.log10(1.0+slf._argflag['generate']['pixelsize']/299792.458))
                        wavein = lwavemin*(1.0+slf._argflag['generate']['pixelsize']/299792.458)**np.arange(npix)
                        fluxin = np.zeros(wavein.size)
                        fluein = np.zeros(wavein.size)
                        contin, zeroin, systin, fitrin, loadin = np.zeros(wavein.size), np.zeros(wavein.size), np.zeros(wavein.size), np.ones(wavein.size), np.ones(wavein.size)
                    elif bntyp == 'A':
                        npix = 1.0 + np.ceil((lwavemax-lwavemin)/slf._argflag['generate']['pixelsize'])
                        wavein = lwavemin + slf._argflag['generate']['pixelsize']*np.arange(npix)
                        fluxin = np.zeros(wavein.size)
                        fluein = np.zeros(wavein.size)
                        contin, zeroin, systin, fitrin, loadin = np.zeros(wavein.size), np.zeros(wavein.size), np.zeros(wavein.size), np.ones(wavein.size), np.ones(wavein.size)
                    elif bntyp == 'Hz':
                        npix = 1.0 + np.ceil((lwavemax - lwavemin) / slf._argflag['generate']['pixelsize'])
                        wavein = lwavemin + slf._argflag['generate']['pixelsize'] * np.arange(npix)
                        fluxin = np.zeros(wavein.size)
                        fluein = np.zeros(wavein.size)
                        contin, zeroin, systin, fitrin, loadin = np.zeros(wavein.size), np.zeros(wavein.size), np.zeros(
                            wavein.size), np.ones(wavein.size), np.ones(wavein.size)
                    else:
                        msgs.error("Sorry, I do not know the bintype: {0:s}".format(bntyp))
            # Is this a onefits file?
            elif slf._isonefits:
                wavein, fluxin, fluein, contin, zeroin, systin, fitrin, loadin = load_fits(slf._argflag['run']['modname'], colspl, wfe, verbose=slf._argflag['out']['verbose'], ext=datnum)
            # Has the user passed in their own data array?
            elif data is not None:
                wavein, fluxin, fluein, contin, zeroin, systin, fitrin, loadin = load_userdata(data, colspl, wfe, verbose=slf._argflag['out']['verbose'])
            # Otherwise, load the data from a file
            else:
                wavein, fluxin, fluein, contin, zeroin, systin, fitrin, loadin = load_datafile(filename, colspl, wfe, verbose=slf._argflag['out']['verbose'], datatype=slf._argflag['run']['datatype'])
        except:
            msgs.error("Error reading in file -"+msgs.newline()+filename)
        # Store the filename for later
        snipnames[sind].append(filename)
        resn[sind] = np.append(resn[sind], tempresn)
        shft[sind] = np.append(shft[sind], tempshft)
        datopt['specid'][sind].append(specidtxt)
        datopt['plotone'][sind].append(setplot)
        datopt['label'][sind].append(uselabel)
        datopt['fitrange'][sind].append(fitrtxt)
        datopt['loadrange'][sind].append(loadrtxt)
        datopt['yrange'][sind].append(yrngtxt)
        datopt['columns'][sind].append(wfe)
        datopt['bintype'][sind].append(bntyp)
        datopt['systematics'][sind].append(systfile)
        if systval is None:
            datopt['systmodule'][sind].append(slf._argflag['sim']['systmodule'])
        elif systval == 'none':
            datopt['systmodule'][sind].append(None)
        else:
            datopt['systmodule'][sind].append(systval)
        # Put the data to be used for the chi-squared into a single array
        if wavemin == -1.0:
            wf = np.arange(wavein.size)
            wavemin, wavemax = np.min(wavein), np.max(wavein)
        elif wavemin == -2.0:
            wf = np.where(fitrin != 0.0)
            wavemin, wavemax = np.min(wavein[wf]), np.max(wavein[wf])
        else:
            wf = np.where((wavein >= wavemin) & (wavein <= wavemax))
        # Specify the loaded data
        if lwavemin == -1.0:
            w  = np.arange(wavein.size)
            lwavemin, lwavemax = np.min(wavein), np.max(wavein)
        elif lwavemin == -2.0:
            fspl = tempresn.strip(')').split('(')
            pars=fspl[1].split(',')
            wli = np.where(loadin==1.0)
            lwavemin, lwavemax = np.min(wavein[wli]), np.max(wavein[wli])
            wvmnres, wvmxres = slf._funcarray[1][fspl[0]].getminmax(slf._funcarray[2][fspl[0]], pars, [wavemin,wavemax])
            tmpmin = min(wvmnres, lwavemin)
            tmpmax = max(wvmxres, lwavemax)
            w  = np.where((wavein >= tmpmin) & (wavein <= tmpmax))
            del wli
        else:
            # Perform a check on the loaded wavelength range
            if lwavemin is None or lwavemax is None:
                lwavemin = wavemin
                lwavemax = wavemax
            if lwavemin > wavemin:
                msgs.warn("The minimum wavelength loaded should be less than the minimum fitted wavelength")
                lwavemin = wavemin
            if lwavemax < wavemax:
                msgs.warn("The maximum wavelength loaded should be greater than the maximum fitted wavelength")
                lwavemax = wavemax
            fspl = tempresn.strip(')').split('(')
            pars=fspl[1].split(',')
            wvmnres, wvmxres = slf._funcarray[1][fspl[0]].getminmax(slf._funcarray[2][fspl[0]], pars, [lwavemin,lwavemax])
            w  = np.where((wavein >= wvmnres) & (wavein <= wvmxres))
#		if loadall: w = np.arange(wavein.size) # If this keyword was set, load all data from this file
        if np.size(w) == 0:
            msgs.error("No data was found within the fitrange limits for the file -"+msgs.newline()+filename)
        datopt['nsubpix'][sind].append( nspix*get_binsize(wavein[w],bintype=bntyp,verbose=slf._argflag['out']['verbose']) )
        posnfit[sind].append(wavemin)
        posnfit[sind].append(wavemax)
        wavefit[sind] = np.append(wavefit[sind], wavein[wf])
        fluxfit[sind] = np.append(fluxfit[sind], fluxin[wf])
        fluefit[sind] = np.append(fluefit[sind], fluein[wf])
        contfit[sind] = np.append(contfit[sind], contin[wf])
        zerofit[sind] = np.append(zerofit[sind], zeroin[wf])
        # Put the full data in separate array
        posnfull[sind].append(wavefull[sind].size)
        wavefull[sind] = np.append(wavefull[sind], wavein[w])
        fluxfull[sind] = np.append(fluxfull[sind], fluxin[w])
        fluefull[sind] = np.append(fluefull[sind], fluein[w])
        contfull[sind] = np.append(contfull[sind], contin[w])
        zerofull[sind] = np.append(zerofull[sind], zeroin[w])
        systfull[sind] = np.append(systfull[sind], systin[w])
        datnum += 1
    # And finally append the total size of the array.
    for i in range(specid.size):
        posnfull[i].append(wavefull[i].size)
    # Update the slf class with the loaded data
    slf._snipnames, slf._resn, slf._shft, slf._datlines, slf._specid, slf._datopt = snipnames, resn, shft, datlines, specid, datopt
    slf._posnfull, slf._posnfit = posnfull, posnfit
    slf._wavefull, slf._fluxfull, slf._fluefull, slf._contfull, slf._zerofull, slf._systfull = wavefull, fluxfull, fluefull, contfull, zerofull, systfull
    slf._wavefit, slf._fluxfit, slf._fluefit, slf._contfit, slf._zerofit = wavefit, fluxfit, fluefit, contfit, zerofit
    msgs.info("Data loaded successfully",verbose=slf._argflag['out']['verbose'])
    return


def load_userdata(data, colspl, wfe, verbose=2):
    wfek = list(wfe.keys())
    usecols=()
    ucind=dict({})
    uccnt=0
    for j in range(len(wfek)):
        if wfe[wfek[j]] == -1: continue
        usecols += (wfe[wfek[j]],)
        ucind[wfek[j]]=uccnt
        uccnt+=1
    datain = data[usecols,:]
    wavein, fluxin, fluein = datain[ucind['wave'],:].astype(np.float64), datain[ucind['flux'],:].astype(np.float64), datain[ucind['error'],:].astype(np.float64)
    ncols = datain.shape[1]
    if len(colspl) > ncols:
        msgs.error("The data only have {0:d} columns. Have you specified too many".format(ncols)+msgs.newline()+"columns with the 'columns' keyword?")
    # Read the continuum data
    if wfe['continuum'] != -1:
        try:
            contin = datain[ucind['continuum'],:]
        except:
            msgs.warn("A continuum was not provided as input", verbose=verbose)
            contin = np.zeros(wavein.size)
    else: contin = np.zeros(wavein.size)
    # Read the zero-level data
    if wfe['zerolevel'] != -1:
        try:
            zeroin = datain[ucind['zerolevel'],:]
        except:
            msgs.warn("A zero-level was not provided as input", verbose=verbose)
            zeroin = np.zeros(wavein.size)
    else: zeroin = np.zeros(wavein.size)
    # Read the systematics information
    if wfe['systematics'] != -1:
        try:
            systin = datain[ucind['systematics'],:]
        except:
            msgs.warn("Systematics information was not provided as input", verbose=verbose)
            systin = np.zeros(wavein.size)
    else: systin = np.zeros(wavein.size)
    # Read the fitrange information
    if wfe['fitrange']  != -1:
        try:
            fitrin = datain[ucind['fitrange'],:]
        except:
            msgs.warn("fitrange information was not provided as input", verbose=verbose)
            fitrin = np.ones(wavein.size).astype(np.int32)
    else: fitrin = np.ones(wavein.size).astype(np.int32)
    # Read the loadrange information
    if wfe['loadrange']  != -1:
        try:
            loadin = datain[ucind['loadrange'],:]
        except:
            msgs.warn("loadrange information was not provided as input", verbose=verbose)
            loadin = np.ones(wavein.size).astype(np.int32)
    else: loadin = np.ones(wavein.size).astype(np.int32)
    # Now return
    return wavein, fluxin, fluein, contin, zeroin, systin, fitrin, loadin


def load_datafile(filename, colspl, wfe, verbose=2, datatype="default"):
    wfek = list(wfe.keys())
    dattyp = filename.split(".")[-1]
    if dattyp in ['dat', 'ascii', 'txt']:
        return load_ascii(filename, colspl, wfe, wfek, verbose=verbose)
    elif dattyp in ['fits','fit']:
        return load_fits(filename, colspl, wfe, verbose=verbose, datatype=datatype)
    else:
        msgs.error("Sorry, I don't know how to read in ."+dattyp+" files.")
    return


def load_ascii(filename, colspl, wfe, wfek, verbose=2):
#	usecols=()
#	ucind=dict({})
#	uccnt=0
#	for j in range(len(wfek)):
#		if wfe[wfek[j]] == -1: continue
#		usecols += (wfe[wfek[j]],)
#		ucind[wfek[j]]=uccnt
#		uccnt+=1
#	try:
#		datain = np.loadtxt(filename, dtype=np.float64, usecols=usecols).transpose()
#	except:
    wavein, fluxin, fluein, contin, zeroin, systin, fitrin, loadin = None, None, None, None, None, None, None, None
    for j in range(len(wfek)):
        if wfe[wfek[j]] == -1: continue
        try:
            onecol = np.loadtxt(filename, dtype=np.float64, usecols=(wfe[wfek[j]],))
            if   wfek[j] == 'wave':        wavein = onecol
            elif wfek[j] == 'flux':        fluxin = onecol
            elif wfek[j] == 'error':       fluein = onecol
            elif wfek[j] == 'continuum':   contin = onecol
            elif wfek[j] == 'zerolevel':   zeroin = onecol
            elif wfek[j] == 'systematics': systin = onecol
            elif wfek[j] == 'fitrange':    fitrin = onecol
            elif wfek[j] == 'loadrange':   loadin = onecol
            else:
                msgs.error("I didn't understand '{0:s}' when reading in -".format(wfek[j])+msgs.newline()+filename)
        except:
            msgs.warn("{0:s} information was not provided as input for the file -".format(wfek[j])+msgs.newline()+filename, verbose=verbose)
            if wfek[j] not in ['systematics']:
                msgs.info("The {0:s} will be output for the above file".format(wfek[j]), verbose=verbose)
    if wavein is None or fluxin is None or fluein is None:
        msgs.error("Wavelength, flux or error array was not provided for -"+msgs.newline()+filename)
    if contin is None: contin = np.zeros(wavein.size)
    if zeroin is None: zeroin = np.zeros(wavein.size)
    if systin is None: systin = np.zeros(wavein.size)
    if fitrin is None: fitrin = np.ones(wavein.size).astype(np.int32)
    if loadin is None: loadin = np.ones(wavein.size).astype(np.int32)
    # Now return
    return wavein, fluxin, fluein, contin, zeroin, systin, fitrin, loadin


def load_fits(filename, colspl, wfe, verbose=2, ext=0, datatype='default'):
    infile = pyfits.open(filename)
    datain=infile[ext].data
    foundtype=False
    # First test what format the data file is
    try: # Is it an ALIS file?
        alisfits = infile[0].header['alisfits']
        if alisfits == "onefits":
            wavein, fluxin, fluein = datain[wfe['wave'],:], datain[wfe['flux'],:], datain[wfe['error'],:]
        elif alisfits == "fits":
            wavein, fluxin, fluein = datain[wfe['wave'],:], datain[wfe['flux'],:], datain[wfe['error'],:]
        foundtype = True
    except:
        pass # This is not an ALIS fits file.
    if not foundtype:
        if datatype.lower() in ['default','uvespopler']:
            try:
                #if wfe['wave'] != "-1": msgs.warn("You shouldn't need to specify the column of 'wave' for a"+msgs.newline()+"'default' fits file", verbose=verbose)
                # Load the wavelength scale
                crvala=infile[ext].header['crval1']
                cdelta=infile[ext].header['cdelt1']
                crpixa=infile[ext].header['crpix1']
                cdelt_alta=infile[ext].header['cd1_1']
                if cdelta == 0.0: cdelta = cdelt_alta
                pixscalea=infile[ext].header['cdelt1']
                wrng = np.arange(datain.shape[1])
                wavein = 10.0**(((wrng - (crpixa - 1))*cdelta)+crvala)
                fluxin, fluein = datain[wfe['flux'],:], datain[wfe['error'],:]
                foundtype = True
            except:
                pass
        elif datatype.lower() in ['hiredux','hiresredux']:
            try:
                #if wfe['wave'] != "-1": msgs.warn("You shouldn't need to specify the column of 'wave' for a"+msgs.newline()+"'HIRESredux' fits file", verbose=verbose)
                # Load the wavelength scale
                crvala=infile[ext].header['crval1']
                cdelta=infile[ext].header['cdelt1']
                pixscalea=infile[ext].header['cdelt1']
                wrng = np.arange(infile[ext].header['naxis1'])
                wavein = 10.0**((wrng*cdelta)+crvala)
                fluxin = datain[:]
                if os.path.exists(".".join(filename.split(".")[:-1])[:-1]+"e." + filename.split(".")[-1]):
                    reduxfn = ".".join(filename.split(".")[:-1])[:-1]+"e." + filename.split(".")[-1]
                elif os.path.exists(".".join(filename.split(".")[:-1])[:-1]+"E." + filename.split(".")[-1]):
                    reduxfn = ".".join(filename.split(".")[:-1])[:-1]+"E." + filename.split(".")[-1]
                else:
                    msgs.error("Unable to load error spectrum for HIREDUX file:"+msgs.newline()+filename)
                intemp = pyfits.open(reduxfn)
                derrin = intemp[ext].data
                fluein = derrin[:]
                foundtype = True
            except:
                pass
        else:
            pass
    if not foundtype:
        msgs.info("An input fits file is not the default format", verbose=verbose)
        msgs.error("Please specify the type of fits file with run+datatype")
    # Read in the other columns of data
    if datatype.lower() not in ["hiresredux","hiredux"]: ncols = datain.shape[1]
    else: ncols = 1
    if len(colspl) > ncols and datatype.lower() not in ["hiresredux","hiredux"]:
        msgs.error("The following file -"+msgs.newline()+filename+msgs.newline()+"only has "+str(ncols)+" columns")
    # Read in the continuum
    if wfe['continuum'] != -1:
        try:
            contin = datain[wfe['continuum'],:].astype(np.float64)
            msgs.warn("Data found in column {0:d} for the file -".format(wfe['continuum']+1)+msgs.newline()+filename+msgs.newline()+"will not be used for the continuum", verbose=verbose)
            contin = np.zeros(wavein.size).astype(np.float64)
        except:
            msgs.warn("A continuum was not provided as input for the file -"+msgs.newline()+filename, verbose=verbose)
            msgs.info("The continuum will be output for the above file", verbose=verbose)
            contin = np.zeros(wavein.size).astype(np.float64)
    else: contin = np.zeros(wavein.size).astype(np.float64)
    # Read in the zero-level
    if wfe['zerolevel'] != -1:
        try:
            zeroin = datain[wfe['zerolevel'],:]
            msgs.warn("Data found in column {0:d} for the file -".format(wfe['zerolevel']+1)+msgs.newline()+filename+msgs.newline()+"will not be used for the zero level", verbose=verbose)
            contin = np.zeros(wavein.size).astype(np.float64)
        except:
            msgs.warn("A zero-level was not provided as input for the file -"+msgs.newline()+filename, verbose=verbose)
            msgs.info("The zero-level will be output for the above file", verbose=verbose)
            zeroin = np.zeros(wavein.size).astype(np.float64)
    else: zeroin = np.zeros(wavein.size).astype(np.float64)
    # Read in the systematics information
    if wfe['systematics'] != -1:
        try:
            systin = datain[wfe['systematics'],:]
        except:
            msgs.warn("Systematics information was not provided as input for the file -"+msgs.newline()+filename, verbose=verbose)
            systin = np.zeros(wavein.size).astype(np.float64)
    else: systin = np.zeros(wavein.size).astype(np.float64)
    # Read in the fit range
    if wfe['fitrange']  != -1:
        try:
            fitrin = datain[wfe['fitrange'],:].astype(np.float64)
        except:
            msgs.warn("A fitting range was not provided as input for the file -"+msgs.newline()+filename, verbose=verbose)
            msgs.info("The fitting range will be output for the above file", verbose=verbose)
            fitrin = np.ones(wavein.size).astype(np.float64)
    else: fitrin = np.ones(wavein.size).astype(np.float64)
    # Read in the load range
    if wfe['loadrange']  != -1:
        try:
            loadin = datain[wfe['loadrange'],:].astype(np.float64)
        except:
            msgs.warn("A loadrange was not provided as input for the file -"+msgs.newline()+filename, verbose=verbose)
            msgs.info("The loadrange will be output for the above file", verbose=verbose)
            loadin = np.ones(wavein.size).astype(np.float64)
    else: loadin = np.ones(wavein.size).astype(np.float64)
    # Now return
    return wavein, fluxin, fluein, contin, zeroin, systin, fitrin, loadin


def load_model(slf, modlines, updateself=True):
    """
    Load the Model
    """
    msgs.info("Loading the model",verbose=slf._argflag['out']['verbose'])
    # Get the calls to each of the functions
    funcused=[]
    emab=[]
    # Set up the dictionary for the model details
    modpass = dict({'mtyp':[],  # Store the type of model
                    'mpar':[],  # Store the parameters for the model
                    'mtie':[],  # Store the tied parameters and what parameter they are tied to.
                    'mlim':[],  # Store the limited parameters and their value
                    'mlnk':[],  # Store the linked parameter strings
                    'mfix':[],  # Determine if a parameter is fixed
                    'tpar':[],  # Sequence of letters used to tie/fix parameters
                    'mkey':[],  # Keyword parameters
                    'psto':[],
                    'p0':[],    # Variables/parameters
                    'emab':[],  # Emission or Absorption?
                    'line':[]}) # The index of modlines that corresponds to this parameter
    cntr=0
    # Load the models form
    parid = []
    pnumlin=[]
    zl_spcfd = False
    for i in range(len(modlines)):
        if len(modlines[i].strip()) == 0: continue # Nothing on a line
        nocoms = modlines[i].lstrip().split('#')[0] # Remove everything on a line after the first instance of a comment symbol: #
        if len(nocoms) == 0: continue # A comment line
        mdlspl = nocoms.split()
        if mdlspl[0] == 'fix' or mdlspl[0] == 'lim': continue # Deal with this later
        if mdlspl[0].strip() == 'emission':
            emabtag = 'em'
            emab.append('em')
            continue
        elif mdlspl[0].strip() == 'absorption':
            emabtag = 'ab'
            emab.append('ab')
            continue
        elif mdlspl[0].strip() == 'zerolevel':
            if zl_spcfd: msgs.error("You can only specify the zero-level once")
            zl_spcfd = True
            emabtag = 'zl'
            emab.append('zl')
            continue
        elif mdlspl[0] in slf._funcarray[0]: # Load the model parameters
            instr = nocoms.strip().lstrip(mdlspl[0]).strip()
            modpass, paridt = slf._funcarray[1][mdlspl[0]].load(slf._funcarray[2][mdlspl[0]], instr, cntr, modpass, slf._specid)
            for j in range(len(pnumlin),len(modpass['p0'])): pnumlin.append(i) # Store which element of modlines each parameter comes from
            parid.append(paridt)
            if mdlspl[0] not in funcused: funcused.append(mdlspl[0])
        else:
            msgs.error("Keyword '"+mdlspl[0]+"' is not a recognised model")
        # Append whether this is emission or absorption and add one to the counter.
        if mdlspl[0].strip() in ['variable','random']:
            modpass['emab'].append('va')
        else:
            modpass['emab'].append(emabtag)
        cntr += 1
    # Load the functional form of the FWHM
    for i in range(len(slf._resn)):
        for j in range(len(slf._resn[i])):
            fspl = slf._resn[i][j].strip(')').split('(')
            modpass, paridt = slf._funcarray[1][fspl[0]].load(slf._funcarray[2][fspl[0]], fspl[1], cntr, modpass, slf._specid)
            modpass['emab'].append('cv')
            parid.append(paridt)
            pnumlin.append("resolution="+slf._resn[i][j])
            cntr += 1
    # Load the functional form of the shift
    for i in range(len(slf._shft)):
        for j in range(len(slf._shft[i])):
            if slf._shft[i][j] == "none":
                fspln, fsplv = "Ashift", "0.0"
                modpass, paridt = slf._funcarray[1][fspln].load(slf._funcarray[2][fspln], fsplv, cntr, modpass, slf._specid, forcefix=True)
                modpass['emab'].append('sh')
                parid.append(paridt)
                #pnumlin.append("shift="+slf._shft[i][j])
            else:
                fspl = slf._shft[i][j].strip(')').split('(')
                modpass, paridt = slf._funcarray[1][fspl[0]].load(slf._funcarray[2][fspl[0]], fspl[1], cntr, modpass, slf._specid)
                modpass['emab'].append('sh')
                parid.append(paridt)
                #pnumlin.append("shift="+slf._shft[i][j])
            cntr += 1
    # Now go through modlines again and make the appropriate changes to the modpass dictionary
    cntr = 0
    fixparid=[]
    limparid=[]
    for i in range(0,len(modlines)):
        if len(modlines[i].strip()) == 0: continue # Nothing on a line
        nocoms = modlines[i].lstrip().split('#')[0] # Remove everything on a line after the first instance of a comment symbol: #
        if len(nocoms) == 0: continue # A comment line
        mdlspl = nocoms.split()
        if mdlspl[0] == 'emission' or mdlspl[0] == 'absorption' or mdlspl[0] == 'zerolevel': continue
        if mdlspl[0] == 'fix':
            if len(mdlspl) != 4: msgs.error("Keyword 'fix' requires three arguments on line -"+msgs.newline()+modlines[i])
            # Make some changes to the function instance based on input file
            if mdlspl[1] == "param": # A given parameter is to have limits
                fixparid.append(mdlspl)
                continue
            else:
                if mdlspl[1] not in slf._funcarray[0]: msgs.error("Keyword 'fix' does not accept the parameter "+mdlspl[1]+" on line -"+msgs.newline()+modlines[i])
                if mdlspl[2] not in slf._funcarray[2][mdlspl[1]]._parid: msgs.error("Keyword "+mdlspl[1]+" does not accept the parameter "+mdlspl[2]+" on line -"+msgs.newline()+modlines[i])
                find = np.where(np.array(slf._funcarray[2][mdlspl[1]]._parid)==mdlspl[2])[0][0]
                if mdlspl[3] in ['None','none','NONE']: slf._funcarray[2][mdlspl[1]]._fixpar[find] = None
                else: slf._funcarray[2][mdlspl[1]]._fixpar[find] = mdlspl[3]
                continue
        elif mdlspl[0] == 'lim':
            if len(mdlspl) != 4: msgs.error("Keyword 'lim' requires three arguments on line -"+msgs.newline()+modlines[i])
            if mdlspl[1] == "param": # A given parameter is to have limits
                limparid.append(mdlspl)
                continue
            else:
                # Make some changes to the function instance based on input file
                if mdlspl[1] not in slf._funcarray[0]: msgs.error("Keyword 'lim' does not accept the parameter "+mdlspl[1]+" on line -"+msgs.newline()+modlines[i])
                if mdlspl[2] not in slf._funcarray[2][mdlspl[1]]._parid: msgs.error("Keyword "+mdlspl[1]+" does not accept the parameter "+mdlspl[2]+" on line -"+msgs.newline()+modlines[i])
                find = np.where(np.array(slf._funcarray[2][mdlspl[1]]._parid)==mdlspl[2])[0][0]
                limspl=mdlspl[3].strip('[]').split(',')
                if len(limspl) != 2: msgs.error("Keyword 'lim' only accepts a two element array on line -"+msgs.newline()+modlines[i])
                for j in range(2):
                    try:
                        if limspl[j].lower() == 'none':
                            slf._funcarray[2][mdlspl[1]]._limited[find][j] = 0
                            slf._funcarray[2][mdlspl[1]]._limits[find][j]  = 0.0
                        else:
                            slf._funcarray[2][mdlspl[1]]._limited[find][j] = 1
                            slf._funcarray[2][mdlspl[1]]._limits[find][j]  = np.float64(limspl[j])
                    except:
                        msgs.error("Keyword 'lim' must be a floating point number or 'None' on line -"+msgs.newline()+modlines[i])
                continue
        # Adjust any parameters for these models
        for j in range(len(parid[cntr])):
            if slf._funcarray[2][mdlspl[0]]._fixpar[parid[cntr][j]] is None: pass
            else: slf._funcarray[1][mdlspl[0]].adjust_fix(slf._funcarray[2][mdlspl[0]], modpass, cntr, j, parid[cntr][j])
            if slf._funcarray[2][mdlspl[0]]._limited[parid[cntr][j]][0] == (0 if modpass['mlim'][cntr][j][0]==None else 1) and slf._funcarray[2][mdlspl[0]]._limited[j][0] == modpass['mlim'][cntr][j][0]: pass
            else: slf._funcarray[1][mdlspl[0]].adjust_lim(slf._funcarray[2][mdlspl[0]], modpass, cntr, j, 0, parid[cntr][j])
            if slf._funcarray[2][mdlspl[0]]._limited[parid[cntr][j]][1] == (0 if modpass['mlim'][cntr][j][1]==None else 1) and slf._funcarray[2][mdlspl[0]]._limited[j][1] == modpass['mlim'][cntr][j][1]: pass
            else: slf._funcarray[1][mdlspl[0]].adjust_lim(slf._funcarray[2][mdlspl[0]], modpass, cntr, j, 1, parid[cntr][j])
        cntr += 1
    # Now go through and adjust the parameters of the instrument resolution
    for i in range(len(slf._resn)):
        for j in range(len(slf._resn[i])):
            fspl = slf._resn[i][j].strip(')').split('(')
            for k in range(len(parid[cntr])):
                if slf._funcarray[2][fspl[0]]._fixpar[parid[cntr][k]] is None: pass
                else: slf._funcarray[1][fspl[0]].adjust_fix(slf._funcarray[2][fspl[0]], modpass, cntr, k, parid[cntr][k])
                if slf._funcarray[2][fspl[0]]._limited[parid[cntr][k]][0] == (None if modpass['mlim'][cntr][k][0]==0 else 1): pass
                else: slf._funcarray[1][fspl[0]].adjust_lim(slf._funcarray[2][fspl[0]], modpass, cntr, k, 0, parid[cntr][k])
                if slf._funcarray[2][fspl[0]]._limited[parid[cntr][k]][1] == (None if modpass['mlim'][cntr][k][1]==0 else 1): pass
                else: slf._funcarray[1][fspl[0]].adjust_lim(slf._funcarray[2][fspl[0]], modpass, cntr, k, 1, parid[cntr][k])
            cntr += 1
    # Now go through and adjust the parameters of the shift
    for i in range(len(slf._shft)):
        for j in range(len(slf._shft[i])):
            if slf._shft[i][j] == "none":
                fspl = ["Ashift", 0.0]
            else:
                fspl = slf._shft[i][j].strip(')').split('(')
            for k in range(len(parid[cntr])):
                if slf._funcarray[2][fspl[0]]._fixpar[parid[cntr][k]] is None: pass
                else: slf._funcarray[1][fspl[0]].adjust_fix(slf._funcarray[2][fspl[0]], modpass, cntr, k, parid[cntr][k])
                if slf._funcarray[2][fspl[0]]._limited[parid[cntr][k]][0] == (None if modpass['mlim'][cntr][k][0]==0 else 1): pass
                else: slf._funcarray[1][fspl[0]].adjust_lim(slf._funcarray[2][fspl[0]], modpass, cntr, k, 0, parid[cntr][k])
                if slf._funcarray[2][fspl[0]]._limited[parid[cntr][k]][1] == (None if modpass['mlim'][cntr][k][1]==0 else 1): pass
                else: slf._funcarray[1][fspl[0]].adjust_lim(slf._funcarray[2][fspl[0]], modpass, cntr, k, 1, parid[cntr][k])
            cntr += 1
    # Globally set the fixed and limited parameter values set by "fix param" and "lim param"
    for i in range(len(fixparid)):
        foundit=False
        tparid=None
        for j in range(len(modpass['tpar'])):
            if modpass['tpar'][j][0]==fixparid[i][2]:
                foundit=True
                tparid=j
                break
        if not foundit: msgs.error("Could not fix parameter with tie id '{0:s}' -- '{0:s}' does not exist:".format(fixparid[i][2])+msgs.newline()+" ".join(fixparid[i]))
        # Adjust modpass to reflect the fixed parameter
        mlev=0
        for m in range(len(modpass['mtie'])):
            for j in range(len(modpass['mtie'][m])):
                if modpass['mtie'][m][j]==tparid:
                    if fixparid[i][3].lower() == 'true': modpass['mfix'][m][j]=1
                    elif fixparid[i][3].lower() == 'false': modpass['mfix'][m][j]=0
                    else:
                        try:
                            modpass['mpar'][m][j]=float(fixparid[i][3])
                            modpass['mfix'][m][j]=1
                        except:
                            msgs.error("I could not fix the parameter value with tie {0:s}".format(fixparid[i][2])+msgs.newline()+"must be one of 'True', 'False', or a floating point number")
                elif mlev==modpass['tpar'][tparid][1]:
                    if fixparid[i][3].lower() == 'true': modpass['mfix'][m][j]=1
                    elif fixparid[i][3].lower() == 'false': modpass['mfix'][m][j]=0
                    else:
                        try:
                            modpass['mpar'][m][j]=float(fixparid[i][3])
                            modpass['p0'][mlev]=float(fixparid[i][3])
                            modpass['mfix'][m][j]=1
                        except:
                            msgs.error("I could not fix the parameter value with tie {0:s}".format(fixparid[i][2])+msgs.newline()+"must be one of 'True', 'False', or a floating point number")
                    mlev += 1
                elif modpass['mtie'][m][j] == -1:
                    mlev += 1
    for i in range(len(limparid)):
        foundit=False
        tparid=None
        for j in range(len(modpass['tpar'])):
            if modpass['tpar'][j][0]==limparid[i][2]:
                foundit=True
                tparid=j
        if not foundit: msgs.error("Could not limit parameter with tied id '{0:s}' -- '{0:s}' does not exist:".format(limparid[i][2])+msgs.newline()+" ".join(limparid[i]))
        limspl=limparid[i][3].strip('[]').split(',')
        # Adjust modpass to reflect the fixed parameter
        mlev=0
        for m in range(len(modpass['mtie'])):
            for j in range(len(modpass['mtie'][m])):
                if modpass['mtie'][m][j]==tparid:
                    # Set lower limit
                    if limspl[0].lower() == 'none': modpass['mlim'][m][j][0]=None
                    else:
                        try:
                            modpass['mlim'][m][j][0]=float(limspl[0])
                        except:
                            msgs.error("I could not limit the lower parameter value with tie {0:s}".format(limparid[i][2])+msgs.newline()+"check the form of the model line")
                    # Set upper limit
                    if limspl[1].lower() == 'none': modpass['mlim'][m][j][1]=None
                    else:
                        try:
                            modpass['mlim'][m][j][1]=float(limspl[1])
                        except:
                            msgs.error("I could not limit the lower parameter value with tie {0:s}".format(limparid[i][2])+msgs.newline()+"check the form of the model line")
                elif mlev==modpass['tpar'][tparid][1]:
                    # Set lower limit
                    if limspl[0].lower() == 'none': modpass['mlim'][m][j][0]=None
                    else:
                        try:
                            modpass['mlim'][m][j][0]=float(limspl[0])
                        except:
                            msgs.error("I could not limit the lower parameter value with tie {0:s}".format(limparid[i][2])+msgs.newline()+"check the form of the model line")
                    # Set upper limit
                    if limspl[1].lower() == 'none': modpass['mlim'][m][j][1]=None
                    else:
                        try:
                            modpass['mlim'][m][j][1]=float(limspl[1])
                        except:
                            msgs.error("I could not limit the lower parameter value with tie {0:s}".format(limparid[i][2])+msgs.newline()+"check the form of the model line")
                    mlev += 1
                elif modpass['mtie'][m][j] == -1:
                    mlev += 1
    msgs.info("Model loaded successfully",verbose=slf._argflag['out']['verbose'])
    if updateself:
        slf._emab = emab
        slf._funcused = funcused
    modpass['line'] = pnumlin
    return modpass


def load_links(slf, lnklines):
    """
    Load the Links
    """
    msgs.info("Loading the links",verbose=slf._argflag['out']['verbose'])
    linka, linkb, linke = [], [], []
    lnkcnt = 0
    for i in range(len(lnklines)):
        if len(lnklines[i].strip()) == 0: continue # Nothing on a line
        nocoms = lnklines[i].lstrip().split('#')[0] # Remove everything on a line after the first instance of a comment symbol: #
        if len(nocoms) == 0: continue # A comment line
        lnkspl = nocoms.split()
        # Check the link form is OK
        if len(lnkspl) < 3:
            msgs.error("You must specify links, with spaces, in the following example form -"+msgs.newline()+"va(vb) = 2.4 + vb")
#		if lnkspl[1] not in ["+","-","*","/"]:
#			msgs.error("Links must only contain one of the following operations: '+', '-', '*', or '/'")
#		if lnkspl[1] not in ["="]:
#			msgs.error("Links must only contain one of the following relational operators: '='")
        # get the variables
        try:
            tL, tR = nocoms.split("=")
            exp = tR.strip()
            varA, tb = tL.strip().split("(")
            varB = tb.rstrip(")").split(",")
            for j in range(len(varB)): varB[j] = varB[j].strip()
            varB = sorted(varB,key=len)[::-1]
            if "numpy" in varB:
                msgs.warn("variable 'numpy' cannot be used in links - this string is reserved for numpy functions")
            elif "np" in varB:
                msgs.warn("variable 'np' cannot be used in links - this string is reserved for numpy functions")
        except:
            msgs.error("You must specify links, with spaces, in the following example form -"+msgs.newline()+"va(vb) = 2.4 + vb")
        if varA in linka:
            msgs.warn("A link for the variable '{0:s}' is specified more than once -".format(varA)+msgs.newline()+lnklines[i].rstrip("\n")+msgs.newline()+"This link will be ignored",verbose=slf._argflag['out']['verbose'])
            continue
        # Check the two link parameters exist in the model
        foundA, foundB = False, [False for all in varB]
        for j in range(len(slf._modpass['tpar'])):
            if slf._modpass['tpar'][j][0] == varA:
                foundA = True
                mtc = 0
                for ka in range(len(slf._modpass['mtie'])):
                    for kb in range(len(slf._modpass['mtie'][ka])):
                        if slf._modpass['mtie'][ka][kb] >= 0:
                            # Tied parameter is not included in tpar[*][1]
                            continue
                        if mtc == slf._modpass['tpar'][j][1]-lnkcnt: # You need to subtract lnkcnt here because modpass['mtie'] is being updated during the for loop and changes a -1 to a more negative number
                            slf._modpass['mtie'][ka][kb] = -2-lnkcnt
                            mtc = -1
                            # elif slf._modpass['mtie'][ka][kb] == slf._modpass['tpar'][j][1]: slf._modpass['mtie'][ka][kb] = -2-lnkcnt
                        elif slf._modpass['mtie'][ka][kb] == mtc and mtc != -1: slf._modpass['mtie'][ka][kb] = -2 - lnkcnt
                        #elif mtc != -1: mtc += 1
                        elif slf._modpass['mtie'][ka][kb] == -1 and mtc != -1: mtc += 1
            for k in range(len(varB)):
                if slf._modpass['tpar'][j][0] == varB[k]:
                    foundB[k] = True
        if not foundA:
            msgs.error("Could not find the link string '{0:s}' in the model".format(lnkspl[0]))
        for k in range(len(foundB)):
            if not foundB[k]: msgs.error("Could not find the link string '{0:s}' in the model".format(varB[k]))
        # If they are good, append them to the list, and add one to the link counter
        linka.append(varA)
        linkb.append(varB)
        linke.append(exp)
        lnkcnt += 1
    # Write a dictionary with the relavant operations
    lnkpass = dict({'opA':linka,     # First tied parameter
                    'opB':linkb,     # Array of linked parameters
                    'exp':linke})    # The relational expression
    msgs.info("Links loaded successfully",verbose=slf._argflag['out']['verbose'])
    return lnkpass


def load_onefits(slf,loadname):
    infile = pyfits.open(loadname)
    # Check this is an ALIS onefits file
    try:
        if infile[0].header['alisfits'] != "onefits": 1/0
    except:
        msgs.error("The input file is a fits file, but not an ALIS onefits file."+msgs.newline()+"ALIS does not accept models with extension *.fits unless it"+msgs.newline()+"is an ALIS onefits file. The offending filename is:"+msgs.newline()+loadname)
    msgs.info("You have input an ALIS onefits file. What would you like to do?", verbose=slf._argflag['out']['verbose'])
    print("")
    print("  (1) Plot the starting model")
    print("  (2) Plot the best-fitting model")
    print("  (3) Print the best-fitting model")
    print("  (4) Re-run the fit")
    print("  (5) Extract the .mod file and the data")
#	print "  (6) Plot the best-fitting model"
#	print "  (7) Plot the best-fitting model"
#	print "  (8) Plot the best-fitting model"
#	print "  (9) Plot the best-fitting model"
    print("")
    ans=0
    while ans not in [1,2,3,4,5]:
        ans = input(msgs.input()+"Please select an option (1,2,3,4,5) - ")
        try:
            ans = int(ans)
        except:
            pass
    print("")
    ###########################
    ###  Plot the best-fitting model
    ###################
    if ans == 1:
        try:
            tparlines, tdatlines, tmodlines, tlnklines = infile[0].header['parlines'], infile[0].header['datlines'], infile[0].header['modlines'], infile[0].header['lnklines']
            parlines = ''.join(chr(int(i)) for i in tparlines.split(","))
            datlines = ''.join(chr(int(i)) for i in tdatlines.split(","))
            modlines = ''.join(chr(int(i)) for i in tmodlines.split(","))
            lnklines = ''.join(chr(int(i)) for i in tlnklines.split(","))
        except:
            msgs.error("The onefits file is corrupt")
        slf._isonefits = True
        return parlines.split("\n")+['plot only True'], datlines.split("\n"), modlines.split("\n"), lnklines.split("\n")
    elif ans == 2:
        # Get the best-fitting output model file
        try:
            toutput = infile[0].header['output']
            output  = ''.join(chr(int(i)) for i in toutput.split(","))
        except:
            msgs.error("The onefits file is corrupt")
        parlines, datlines, modlines, lnklines = load_input(slf,textstr=output)
        slf._isonefits = True
        return parlines+['plot only True'], datlines, modlines, lnklines
    ###########################
    ###  Print the best-fitting model
    ###################
    elif ans == 3:
        try:
            toutput = infile[0].header['output']
            output  = ''.join(chr(int(i)) for i in toutput.split(","))
        except:
            msgs.error("The onefits file is corrupt")
        print(output)
        sys.exit()
    ###########################
    ###  Re-run the fit
    ###################
    elif ans == 4:
        # Get the input model file
        try:
            tparlines, tdatlines, tmodlines, tlnklines = infile[0].header['parlines'], infile[0].header['datlines'], infile[0].header['modlines'], infile[0].header['lnklines']
            parlines = ''.join(chr(int(i)) for i in tparlines.split(","))
            datlines = ''.join(chr(int(i)) for i in tdatlines.split(","))
            modlines = ''.join(chr(int(i)) for i in tmodlines.split(","))
            lnklines = ''.join(chr(int(i)) for i in tlnklines.split(","))
        except:
            msgs.error("The onefits file is corrupt")
        slf._isonefits = True
        return parlines.split("\n"), datlines.split("\n"), modlines.split("\n"), lnklines.split("\n")
    ###########################
    ###  Extract the .mod file and the data
    ###################
    elif ans == 5:
        tdirname = ".".join(loadname.split(".")[:-1])+"_dir"
        dirname = tdirname
        numtry=0
        while os.path.exists(dirname):
            dirname = tdirname + "{0:d}".format(numtry)
            numtry += 1
        os.mkdir(dirname)
        os.mkdir(dirname+"/model")
        os.mkdir(dirname+"/data")
        # Extract the data
        datnames=[]
        for i in range(1,infile[0].header['numext']):
            tname = infile[i].header['filename'].split("/")[-1]
            datnames += [".".join(tname.split(".")[:-1])+".dat"]
            np.savetxt(dirname+"/data/"+datnames[-1],np.transpose((infile[i].data)))
        # Extract the starting model
        try:
            tparlines, tdatlines, tmodlines, tlnklines = infile[0].header['parlines'], infile[0].header['datlines'], infile[0].header['modlines'], infile[0].header['lnklines']
            parlines = ''.join(chr(int(i)) for i in tparlines.split(","))
            datlines = ''.join(chr(int(i)) for i in tdatlines.split(","))
            modlines = ''.join(chr(int(i)) for i in tmodlines.split(","))
            lnklines = ''.join(chr(int(i)) for i in tlnklines.split(","))
        except:
            msgs.error("The onefits file is corrupt")
        fmod = open(dirname+"/model/"+infile[0].header['modname'],'w')
        fmod.write(parlines)
        fmod.write("\ndata read\n")
        # Change the name of the files in datlines
        tdatlines = datlines.split("\n")
        numdat=0
        newdatlines=[]
        for i in range(len(tdatlines)):
            if len(tdatlines[i].strip()) == 0 : continue # Nothing on a line
            datspl = tdatlines[i].split()
            datspl[0] = "../data/{0:s}".format(datnames[numdat])
            numdat += 1
            newdatlines += ["  ".join(datspl)]
        fmod.write("\n".join(newdatlines))
        fmod.write("\ndata end\n\nmodel read\n")
        fmod.write(modlines)
        fmod.write("model end\n")
        if lnklines != '':
            fmod.write("\nlink read\n")
            fmod.write(lnklines)
            fmod.write("link end\n")
        msgs.info("The model and data file were extracted into the directory:"+msgs.newline()+dirname,verbose=slf._argflag['out']['verbose'])
        sys.exit()
    else:
        msgs.error("Sorry - option {0:d} is not implemented yet".format(ans))


def load_subpixels(slf, parin):
    nexbins = []
    modtyp=[[] for all in slf._posnfull]
    shind = np.where(np.array(slf._modpass['emab'])=='sh')[0][0]
    # Find narrowest profile in the model
    for sp in range(len(slf._posnfull)):
        lastemab, iea = ['' for all in slf._posnfull[sp][:-1]], [-1 for all in slf._posnfull[sp][:-1]]
        nexbins.append([slf._argflag['run']['nsubmin'] for all in slf._posnfull[sp][:-1]])
        for sn in range(len(slf._posnfull[sp])-1):
            ll = slf._posnfull[sp][sn]
            lu = slf._posnfull[sp][sn+1]
            shmtyp = slf._modpass['mtyp'][shind]
            slf._funcarray[2][shmtyp]._keywd = slf._modpass['mkey'][shind]
            shparams = slf._funcarray[1][shmtyp].set_vars(slf._funcarray[2][shmtyp], parin, slf._levadd[shind], slf._modpass, shind)
            wvrngt = slf._funcarray[1][shmtyp].call_CPU(slf._funcarray[2][shmtyp], slf._wavefull[sp][ll:lu], shparams)
            shind += 1
            wvrng = [wvrngt.min(),wvrngt.max()]
#			wvrng = [slf._wavefull[sp][ll:lu].min(),slf._wavefull[sp][ll:lu].max()]
            modtyp[sp].append([])
            for i in range(0,len(slf._modpass['mtyp'])):
                if slf._modpass['emab'][i] in ['cv','va','sh']: continue # This is a convolution or variable (not emission or absorption)
                if slf._specid[sp] not in slf._modpass['mkey'][i]['specid']: continue # Don't apply this model to this data
                if lastemab[sn] == '' and slf._modpass['emab'][i] != 'em':
                    msgs.error("Model for specid="+slf._snipid[sp]+" must specify emission before absorption")
                if lastemab[sn] != slf._modpass['emab'][i]:
                    modtyp[sp][sn].append(np.array(['']))
                    iea[sn] += 1
                    lastemab[sn] = slf._modpass['emab'][i]
                mtyp=slf._modpass['mtyp'][i]
                if np.where(mtyp==modtyp[sp][sn][iea[sn]])[0].size != 1:
                    modtyp[sp][sn][iea[sn]] = np.append(modtyp[sp][sn][iea[sn]],mtyp)
                    if modtyp[sp][sn][iea[sn]][0] == '': modtyp[sp][sn][iea[sn]] = np.delete(modtyp[sp][sn][iea[sn]], 0)
                mid = np.where(mtyp==modtyp[sp][sn][iea[sn]])[0][0]
                slf._funcarray[2][mtyp]._keywd = slf._modpass['mkey'][i]
                params, nbn = slf._funcarray[1][mtyp].set_vars(slf._funcarray[2][mtyp], parin, slf._levadd[i], slf._modpass, i, wvrng=wvrng, spid=slf._specid[sp], levid=slf._levadd, nexbin=[slf._datopt['bintype'][sp][sn],slf._datopt['nsubpix'][sp][sn]])
                if len(params) == 0: continue
                if nbn > nexbins[sp][sn]:
                    if nbn > slf._argflag['run']['nsubmax']:
                        nexbins[sp][sn] = slf._argflag['run']['nsubmax']
                    else:
                        nexbins[sp][sn] = nbn
    # Find narrowest profile in the instrumental FWHM
    stf, enf = [0 for all in slf._posnfull], [0 for all in slf._posnfull]
    cvind = np.where(np.array(slf._modpass['emab'])=='cv')[0][0]
    for sp in range(len(slf._posnfull)):
        for sn in range(len(slf._posnfull[sp])-1):
            mtyp = slf._modpass['mtyp'][cvind]
            params, nbn = slf._funcarray[1][mtyp].set_vars(slf._funcarray[2][mtyp], parin, slf._levadd[cvind], slf._modpass, cvind, nexbin=[slf._datopt['bintype'][sp][sn],slf._datopt['nsubpix'][sp][sn]])
            if nbn > nexbins[sp][sn]:
                if nbn > slf._argflag['run']['nsubmax']:
                    nexbins[sp][sn] = slf._argflag['run']['nsubmax']
                else:
                    nexbins[sp][sn] = nbn
            cvind += 1
    wavespx, contspx, zerospx, posnspx = [np.array([]) for all in slf._posnfull], [np.array([]) for all in slf._posnfull], [np.array([]) for all in slf._posnfull], [[] for all in slf._posnfull]
    for sp in range(len(slf._posnfull)):
        for sn in range(len(slf._posnfull[sp])-1):
            if nexbins[sp][sn] > slf._argflag['run']['warn_subpix']: msgs.warn("sub-pixellation scale ({0:d}) has exceeded the warning level of: {1:d}".format(nexbins[sp][sn],slf._argflag['run']['warn_subpix']),verbose=slf._argflag['out']['verbose'])
            ll = slf._posnfull[sp][sn]
            lu = slf._posnfull[sp][sn+1]
            binsize=get_binsize(slf._wavefull[sp][ll:lu], bintype=slf._datopt['bintype'][sp][sn], maxonly=False, verbose=slf._argflag['out']['verbose'])
            binlen = 1.0/np.float64(nexbins[sp][sn])
            if slf._datopt['bintype'][sp][sn] == "km/s":
                interpwav = (1.0+((np.arange(nexbins[sp][sn])-(0.5*(nexbins[sp][sn]-1.0)))[np.newaxis,:]*binlen*binsize[:,np.newaxis]/2.99792458E5))
                wavs = (slf._wavefull[sp][ll:lu].reshape(lu-ll,1)*interpwav).flatten(0)
            elif slf._datopt['bintype'][sp][sn] == "A":
                interpwav = ((np.arange(nexbins[sp][sn])-(0.5*(nexbins[sp][sn]-1.0)))[np.newaxis,:]*binlen*binsize[:,np.newaxis])
                wavs = (slf._wavefull[sp][ll:lu].reshape(lu-ll,1) + interpwav).flatten(0)
            elif slf._datopt['bintype'][sp][sn] == "Hz":
                binlen = 1.0 / np.float64(nexbins[sp][sn])
                interpwav = ((np.arange(nexbins[sp][sn]) - (0.5 * (nexbins[sp][sn] - 1.0)))[np.newaxis, :] * binlen * binsize[:,np.newaxis])
                wavs = (slf._wavefull[sp][ll:lu].reshape(lu-ll,1) + interpwav).flatten(0)
            else: msgs.bug("Bintype "+slf._datopt['bintype'][sp][sn]+" is unknown",verbose=slf._argflag['out']['verbose'])
            posnspx[sp].append(wavespx[sp].size)
            wavespx[sp] = np.append(wavespx[sp], wavs)
            if np.all(1.0-slf._contfull[sp][ll:lu]): # No continuum is provided -- no interpolation is necessary
                contspx[sp] = np.append(contspx[sp], np.zeros(np.size(wavs)))
            else: # Do linear interpolation
                gradA = np.append((slf._contfull[sp][ll+1:lu]-slf._contfull[sp][ll:lu-1])/(slf._wavefull[sp][ll+1:lu]-slf._wavefull[sp][ll:lu-1]),(slf._contfull[sp][lu-1]-slf._contfull[sp][lu-2])/(slf._wavefull[sp][lu-1]-slf._wavefull[sp][lu-2])).reshape(lu-ll,1)
                gradB = np.append( np.array([(slf._contfull[sp][ll+1]-slf._contfull[sp][ll])/(slf._wavefull[sp][ll+1]-slf._wavefull[sp][ll])]), (slf._contfull[sp][ll+1:lu]-slf._contfull[sp][ll:lu-1])/(slf._wavefull[sp][ll+1:lu]-slf._wavefull[sp][ll:lu-1]),np.array([(slf._contfull[sp][lu-1]-slf._contfull[sp][lu-2])/(slf._wavefull[sp][lu-1]-slf._wavefull[sp][lu-2])])).reshape(lu-ll,1)
                gradv = np.mean(np.array([gradA,gradB]),axis=0)
                contspx[sp] = np.append(contspx[sp], (slf._contfull[sp][ll:lu].reshape(lu-ll,1) + (wavs.reshape(lu-ll,nexbins[sp][sn])-slf._wavefull[sp][ll:lu].reshape(lu-ll,1))*gradv).flatten(0))
            if np.all(1.0-slf._zerofull[sp][ll:lu]): # No zero-level is provided -- no interpolation is necessary
                zerospx[sp] = np.append(zerospx[sp], np.zeros(np.size(wavs)))
            else: # Do linear interpolation
                gradA = np.append((slf._zerofull[sp][ll+1:lu]-slf._zerofull[sp][ll:lu-1])/(slf._wavefull[sp][ll+1:lu]-slf._wavefull[sp][ll:lu-1]),(slf._zerofull[sp][lu-1]-slf._zerofull[sp][lu-2])/(slf._wavefull[sp][lu-1]-slf._wavefull[sp][lu-2])).reshape(lu-ll,1)
                gradB = np.append( np.array([(slf._zerofull[sp][ll+1]-slf._zerofull[sp][ll])/(slf._wavefull[sp][ll+1]-slf._wavefull[sp][ll])]), (slf._zerofull[sp][ll+1:lu]-slf._zerofull[sp][ll:lu-1])/(slf._wavefull[sp][ll+1:lu]-slf._wavefull[sp][ll:lu-1]),np.array([(slf._zerofull[sp][lu-1]-slf._zerofull[sp][lu-2])/(slf._wavefull[sp][lu-1]-slf._wavefull[sp][lu-2])])).reshape(lu-ll,1)
                gradv = np.mean(np.array([gradA,gradB]),axis=0)
                zerospx[sp] = np.append(zerospx[sp], (slf._zerofull[sp][ll:lu].reshape(lu-ll,1) + (wavs.reshape(lu-ll,nexbins[sp][sn])-slf._wavefull[sp][ll:lu].reshape(lu-ll,1))*gradv).flatten(0))
        posnspx[sp].append(wavespx[sp].size)
#	slf._wavespx, slf._posnspx = wavespx, posnspx
#	slf._nexbins = nexbins
    return wavespx, contspx, zerospx, posnspx, nexbins


def load_par_influence(slf, parin):
    def inlinks(ivar):
        foundit = False
        for i in range(len(slf._modpass['tpar'])):
            if slf._modpass['tpar'][i][0] in slf._links['opA'] and slf._modpass['tpar'][i][1] == ivar: foundit = True
        return foundit
    ##################
    pinfl = []
    uinfl = np.array([],dtype=np.int)
    modtyp=[[] for all in slf._posnfull]
    # Find the model parameters that influence a given sp+sn
    shind = np.where(np.array(slf._modpass['emab'])=='sh')[0][0]
    for sp in range(0,len(slf._posnfull)):
        lastemab, iea = ['' for all in slf._posnfull[sp][:-1]], [-1 for all in slf._posnfull[sp][:-1]]
        pinfl.append([np.array([],dtype=np.int) for all in slf._posnfull[sp][:-1]])
        for sn in range(len(slf._posnfull[sp])-1):
            ll = slf._posnfull[sp][sn]
            lu = slf._posnfull[sp][sn+1]
            shmtyp = slf._modpass['mtyp'][shind]
            slf._funcarray[2][shmtyp]._keywd = slf._modpass['mkey'][shind]
            shparams = slf._funcarray[1][shmtyp].set_vars(slf._funcarray[2][shmtyp], parin, slf._levadd[shind], slf._modpass, shind)
            wvrngt = slf._funcarray[1][shmtyp].call_CPU(slf._funcarray[2][shmtyp], slf._wavefull[sp][ll:lu], shparams)
            shind += 1
            wvrng = [wvrngt.min(),wvrngt.max()]
#			wvrng = [slf._wavefull[sp][ll:lu].min(),slf._wavefull[sp][ll:lu].max()]
            modtyp[sp].append([])
            for i in range(0,len(slf._modpass['mtyp'])):
                if slf._modpass['emab'][i] in ['cv','sh']: continue # This is a convolution or a shift (not emission or absorption)
                if slf._specid[sp] not in slf._modpass['mkey'][i]['specid']: continue # Don't apply this model to this data
                if lastemab[sn] == '' and slf._modpass['emab'][i] != 'em':
                    if slf._modpass['emab'][i] != 'va':
                        msgs.error("Model for specid="+slf._snipid[sp]+" must specify emission before absorption")
                if lastemab[sn] != slf._modpass['emab'][i] and slf._modpass['emab'][i] != 'va':
                    modtyp[sp][sn].append(np.array(['']))
                    iea[sn] += 1
                    lastemab[sn] = slf._modpass['emab'][i]
                mtyp=slf._modpass['mtyp'][i]
                if np.where(mtyp==modtyp[sp][sn][iea[sn]])[0].size != 1:
                    modtyp[sp][sn][iea[sn]] = np.append(modtyp[sp][sn][iea[sn]],mtyp)
                    if modtyp[sp][sn][iea[sn]][0] == '': modtyp[sp][sn][iea[sn]] = np.delete(modtyp[sp][sn][iea[sn]], 0)
                mid = np.where(mtyp==modtyp[sp][sn][iea[sn]])[0][0]
                slf._funcarray[2][mtyp]._keywd = slf._modpass['mkey'][i]
                params, infv = slf._funcarray[1][mtyp].set_vars(slf._funcarray[2][mtyp], parin, slf._levadd[i], slf._modpass, i, wvrng=wvrng, spid=slf._specid[sp], levid=slf._levadd, getinfl=True)
                for j in infv:
                    if not inlinks(j): pinfl[sp][sn] = np.append(pinfl[sp][sn],j)
                    if j not in uinfl and not inlinks(j): uinfl = np.append(uinfl,j)
    # Test if the convolution influences the result of a given sp+sn
    stf, enf = [0 for all in slf._posnfull], [0 for all in slf._posnfull]
    cvind = np.where(np.array(slf._modpass['emab'])=='cv')[0][0]
    for sp in range(len(slf._posnfull)):
        for sn in range(len(slf._posnfull[sp])-1):
            mtyp = slf._modpass['mtyp'][cvind]
            params, infv = slf._funcarray[1][mtyp].set_vars(slf._funcarray[2][mtyp], parin, slf._levadd[cvind], slf._modpass, cvind, getinfl=True)
            cvind += 1
            for j in infv:
                if j not in pinfl[sp][sn] and not inlinks(j): pinfl[sp][sn] = np.append(pinfl[sp][sn],j)
                if j not in uinfl and not inlinks(j): uinfl = np.append(uinfl,j)
    # Test if a shift influences the result of a given sp+sn
    shind = np.where(np.array(slf._modpass['emab'])=='sh')[0][0]
    for sp in range(len(slf._posnfull)):
        for sn in range(len(slf._posnfull[sp])-1):
            mtyp = slf._modpass['mtyp'][shind]
            params, infv = slf._funcarray[1][mtyp].set_vars(slf._funcarray[2][mtyp], parin, slf._levadd[shind], slf._modpass, shind, getinfl=True)
            shind += 1
            for j in infv:
                if j not in pinfl[sp][sn] and not inlinks(j): pinfl[sp][sn] = np.append(pinfl[sp][sn],j)
                if j not in uinfl and not inlinks(j): uinfl = np.append(uinfl,j)
    # Sort the list of all free parameter ID numbers
    uinfl.sort()
    # Map the parameter ID numbers to a *free* parameter ID number
    opinfl = copy.deepcopy(pinfl)
    for sp in range(len(pinfl)):
        for sn in range(len(pinfl[sp])):
            for j in range(len(pinfl[sp][sn])):
                opinfl[sp][sn][j] = np.argwhere(uinfl==pinfl[sp][sn][j])[0][0]
######
    # TEMPORARY DURING TESTING ---- THIS MUST BE REMOVED IN ORDER FOR INFLUENCE TO WORK
#	insrt = range(len(unq_num))
#	for sp in range(len(pinfl)):
#		for sn in range(len(pinfl[sp])):
#			opinfl[sp][sn] = insrt
######
#	print "#################"
#	print pinfl
#	print opinfl
#	sys.exit()
#	print "#################"
#	return opinfl
    return [opinfl,pinfl]


def get_binsize(wave, bintype="km/s", maxonly=True, verbose=2):
    binsize  = np.zeros((2,wave.size))
    binsizet = wave[1:] - wave[:-1]
    if bintype == "km/s": binsizet *= 2.99792458E5/wave[:-1]
    elif bintype == "A" : pass
    elif bintype == "Hz" : pass
    else: msgs.bug("Bintype "+bintype+" is unknown",verbose=verbose)
    maxbin  = np.max(binsizet)
    binsize[0,:-1], binsize[1,1:] = binsizet, binsizet
    binsize[0,-1], binsize[1,0] = maxbin, maxbin
    binsize = binsize.min(0)
    if maxonly: return np.max(binsize)
    else: return binsize


def load_parinfo(slf):
    # Set up parameter dictionary
    parbase={'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.], 'mpside':2, 'step':0}
    parinfo=[]
    for i in range(len(slf._modpass['p0'])):
        parinfo.append(copy.deepcopy(parbase))
        parinfo[i]['value']=slf._modpass['p0'][i]
    # Go through all of the models and place appropriate limits
    level = 0 # A way to locate the first parameter for each part of the model
    levadd = []
    for i in range(len(slf._modpass['mtyp'])):
        levadd.append(level)
        mtyp = slf._modpass['mtyp'][i]
        parinfo, add = slf._funcarray[1][mtyp].set_pinfo(slf._funcarray[2][mtyp], parinfo, level, slf._modpass, slf._links, i)
        level += add
    # Test if some parameters are outside the limits
    if slf._argflag['run']['limpar']:
        for i in range(len(slf._modpass['p0'])):
            if ((parinfo[i]['limited'][0] == 1) & (parinfo[i]['value'] < parinfo[i]['limits'][0])):
                newval = parinfo[i]['limits'][0]
            elif ((parinfo[i]['limited'][1] == 1) & (parinfo[i]['value'] > parinfo[i]['limits'][1])):
                newval = parinfo[i]['limits'][1]
            else:
                continue
            if type(slf._modpass['line'][i]) is int:
                msgs.warn("A parameter that = {0:f} is not within specified limits on line -".format(slf._modpass['p0'][i])+msgs.newline()+slf._modlines[slf._modpass['line'][i]],verbose=slf._argflag['out']['verbose'])
                msgs.info("Setting this parameter to the limiting value of the model: {0:f}".format(newval),verbose=slf._argflag['out']['verbose'])
            else:
                msgs.warn("A parameter that = {0:f} is not within specified limits on line -".format(slf._modpass['p0'][i])+msgs.newline()+slf._modpass['line'][i],verbose=slf._argflag['out']['verbose'])
                msgs.info("Setting this parameter to the limiting value of the model: {0:f}".format(newval),verbose=slf._argflag['out']['verbose'])
            slf._modpass['p0'][i], parinfo[i]['value'] = newval, newval
    return parinfo, levadd


def load_tied(p, ptied=None, infl=None):
    ###############
    if infl is None:
        for i in range(len(ptied)):
            if ptied[i] == '':
                continue
            cmd = 'p[' + str(i) + '] = ' + ptied[i]
            try:
                namespace = dict({'p':p})
                exec(cmd, namespace)
                p = namespace['p']
            except:
                msgs.error("Unable to set a tied parameter to the expression:"+msgs.newline()+ptied[i]+msgs.newline()+"There may be an undefined variable in the links")
    else:
        for i in range(len(ptied)):
            if ptied[i] == '':
                continue
            ival, pstr = getis(ptied[i], i, infl)
            cmd = 'p[' + str(ival) + '] = ' + pstr
            try:
                namespace = dict({'p':p})
                exec(cmd, namespace)
                p = namespace['p']
            except:
                msgs.error("Unable to set a tied parameter to the expression:"+msgs.newline()+ptied[i]+msgs.newline()+"There may be an undefined variable in the links")
    return p


def getis(string, ival, infl, retlhs=True):
    strspl = (" "+string).split("p[")
    ids = [int(i.split("]")[0]) for i in strspl[1:]]
    rids = [-1 for i in strspl[1:]]
    jval = -1
    foundall = False
    for sp in range(len(infl[0])):
        for sn in range(len(infl[0][sp])):
            for j in range(len(infl[0][sp][sn])):
                if infl[1][sp][sn][j] == ival:
                    jval = infl[0][sp][sn][j]
                    if -1 not in rids:
                        foundall = True
                        break
                else:
                    for i in range(len(ids)):
                        if infl[1][sp][sn][j] == ids[i]:
                            rids[i] = infl[0][sp][sn][j]
                            if -1 not in rids and jval != -1:
                                foundall = True
                                break
                if foundall: break
            if foundall: break
        if foundall: break
    # Reassemble the function string
    pstr = strspl[0]
    for i in range(1,len(strspl)):
        pstr += "p["+str(rids[i-1])+"]"+"]".join(strspl[i].split("]")[1:])
    if retlhs: return jval, pstr.strip()
    else: return pstr.strip()
