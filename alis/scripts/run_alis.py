""" Absorption LIne Software
"""
from __future__ import absolute_import, division, print_function

# Import standard libraries
import os
import sys
import traceback
# Import a Chi-Squared minimisation package
# Import useful ALIS definitions:
try:
    from xastropy.xutils import xdebug as debugger
except:
    import pdb as debugger

def parser(options=None):
    import argparse
    from alis import alload

    parser = argparse.ArgumentParser(description=alload.usage('ALIS'),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("alis_modfile", type=str, help="ALIS mod file (.mod extension)")
    parser.add_argument("-c", "--cpus", type=int, help="(all) Number of cpu cores to use")
    parser.add_argument("-g", "--gpu", default=False, help="enable GPU multiprocessing", action="store_true")
    parser.add_argument("-p", "--plot", type=str, 
                        help="({0:s}) plot model fits with MxN panels,\n".format('3x3')+"for no screen plots use: -p 0")
    parser.add_argument("-x", "--xaxis", type=int,
                        help="(0) Plot observed/rest wavelength [0/1] or velocity [2]")
    parser.add_argument("-j", "--justplot", default=False, action="store_true",
                        help="Plot the data and input model - don't fit")
    parser.add_argument("-l", "--labels", default=False, help="Label the absorption components when plotting", action="store_true")
    parser.add_argument("-v", "--verbose", type=int, help="(2) Level of verbosity (0-2)")
    parser.add_argument("-r", "--random", type=int, help="Number of random simulations to perform")
    parser.add_argument("-s", "--startid", type=int, help="Starting ID for the simulations")
    parser.add_argument("-f", "--fits", default=False, help="Write model fits to *.dat files", action="store_true")
    parser.add_argument("-m", "--model", default=False, help="Model", action="store_true")
    parser.add_argument("-w", "--writeover", default=False, help="Clobber", action="store_true")
    #parser.add_argument("-q", "--quick", default=False, help="Quick reduction", action="store_true")
    #parser.print_help()

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    #
    return pargs

def main(args):
    from alis import almsgs
    from alis import alload
    from alis import alis
    msgs = almsgs.msgs()

    # argflag

    debug = True # There are two instances of this (one is in alis just above)
    if debug:
        msgs.work("Read in resolution from column of data",verbose=2)
        msgs.work("With voigt function, if the user says to put an O I profile in specid A, make sure there is actually an O I line in specid A.",verbose=2)
        msgs.work("Prepare a separate .py file for user-created functions",verbose=2)
        msgs.work("Assign a number to every warning and error -- describe this in the manual",verbose=2)
        msgs.work("If emission is not specified for a specid before absorption (in a model with several specid's), the specid printed as an error is always one before",verbose=2)
        argflag = alload.optarg(os.path.realpath(__file__), argv=args)
        # Assign filelist:
#       if sys.argv[-1].split('.')[-1] != 'mod': alload.usage(argflag)
#       else:
        argflag['run']['modname'] = sys.argv[-1]
        alis.ClassMain(argflag)
    else:
        try:
            argflag = alis.alload.optarg(os.path.realpath(__file__), argv=args)
            # Assign filelist:
#			if sys.argv[-1].split('.')[-1] != 'mod': alload.usage(argflag)
#			else:
            argflag['run']['modname'] = sys.argv[-1]
            alis.ClassMain(argflag)
        except Exception:
            # There is a bug in the code, print the file and line number of the error.
            et, ev, tb = sys.exc_info()
            while tb:
                co = tb.tb_frame.f_code
                filename = str(co.co_filename)
                line_no =  str(traceback.tb_lineno(tb))
                tb = tb.tb_next
            filename=filename.split('/')[-1]
            msgs.bug("There appears to be a bug on Line "+line_no+" of "+filename+" with error:"+msgs.newline()+str(ev)+msgs.newline()+"---> please contact the author",verbose=2)
        except SystemExit:
            # The code has found an error in the user input and terminated early.
            pass
        except:
            msgs.bug("There appears to be an undefined (and therefore unhelpful) bug"+msgs.newline()+"---> please contact the author",verbose=2)

