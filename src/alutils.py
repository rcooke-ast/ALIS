"""
This file contains a number of useful utilities
that are used by ALIS.
"""
import almsgs
msgs=almsgs.msgs()

def getreason(idnum,verbose=2):
	if   idnum == 1: return "Both actual and predicted relative reductions in the sum of squares are at most ftol"
	elif idnum == 2: return "Relative error between two consecutive iterates is at most xtol"
	elif idnum == 3: return "Both actual and predicted relative reductions in the sum of squares are at most ftol and the relative error between two consecutive iterates is at most xtol"
	elif idnum == 4: return "The cosine of the angle between fvec and any column of the jacobian is at most gtol in absolute value"
	elif idnum == 5: return "The maximum number of iterations has been reached"
	elif idnum == 6: return "ftol is too small. No further reduction in the sum of squares is possible"
	elif idnum == 7: return "xtol is too small. No further improvement in the approximate solution x is possible"
	elif idnum == 8: return "gtol is too small. fvec is orthogonal to the columns of the jacobian to machine precision"
	elif idnum == 9: return "The relative reduction in the sum of squares is less than atol"
	else:
		msgs.bug("Convergence reason is unknown (probably failed) --- please contact the author",verbose=verbose)
		return "Convergence reason is unknown (probably failed) --- please contact the author"
