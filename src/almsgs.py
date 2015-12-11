import sys

class Colors:
    """
    Create coloured text for messages printed to screen.

    For further details on colours see the following example:
    http://ascii-table.com/ansi-escape-sequences.php
    """
    # Start and end coloured text
    start = "\x1B["
    end   = "\x1B[" + "0m"
    # Clear Backgrounds
    black_CL  = "1;30m"
    blue_CL   = "1;34m"
    green_CL  = "1;32m"
    red_CL    = "1;31m"
    purple_CL = "1;35m"
    # Coloured Backgrounds
    white_RD = "1;37;41m"
    white_GR = "1;37;42m"
    white_BK = "1;37;40m"
    white_BL = "1;37;44m"

    def disable(self):
        self.black_CL = ''
        self.red_CL   = ''
        self.green_CL = ''
        self.black_RD = ''
        self.black_GR = ''

class msgs:
    def __init__(self):
        self._verbose = -1
#		return

    def alisheader(self, prognm, verbose=2):
        header = "##  "
        header += Colors.start + Colors.white_GR + "ALIS : "
        header += "Absorption LIne Software v1.0" + Colors.end + "\n"
        header += "##  "
        header += "Usage : "
        header += "python %s [options] model.mod" % (prognm)
        return header

    def signal_handler(self, signalnum, handler):
        if signalnum == 2:
            self.info("Ctrl+C was pressed. Ending processes...")
            sys.exit()

    def error(self, msg, verbose=None):
        premsg="\n"+Colors.start + Colors.white_RD + "[ERROR]   ::" + Colors.end + " "
        print >>sys.stderr,premsg+msg
        sys.exit()

    def info(self, msg, verbose=None):
        if verbose is None and self._verbose != -1:
            premsg=Colors.start + Colors.green_CL + "[INFO]    ::" + Colors.end + " "
            print >>sys.stderr,premsg+msg
        elif verbose is not None and verbose != -1:
            premsg=Colors.start + Colors.green_CL + "[INFO]    ::" + Colors.end + " "
            print >>sys.stderr,premsg+msg

    def warn(self, msg, verbose=None):
        if verbose is None and self._verbose != -1:
            premsg=Colors.start + Colors.red_CL   + "[WARNING] ::" + Colors.end + " "
            print >>sys.stderr,premsg+msg
        elif verbose is not None and verbose != -1:
            premsg=Colors.start + Colors.red_CL   + "[WARNING] ::" + Colors.end + " "
            print >>sys.stderr,premsg+msg

    def test(self, msg, verbose=None):
        if verbose is None and self._verbose != -1:
            premsg=Colors.start + Colors.white_BL   + "[TEST]    ::" + Colors.end + " "
            print >>sys.stderr,premsg+msg
        elif verbose is not None and verbose != -1:
            premsg=Colors.start + Colors.white_BL   + "[TEST]    ::" + Colors.end + " "
            print >>sys.stderr,premsg+msg

    def bug(self, msg, verbose=None):
        if verbose is None and self._verbose != -1:
            premsg=Colors.start + Colors.white_BK   + "[BUG]     ::" + Colors.end + " "
            print >>sys.stderr,premsg+msg
        elif verbose is not None and verbose != -1:
            premsg=Colors.start + Colors.white_BK   + "[BUG]     ::" + Colors.end + " "
            print >>sys.stderr,premsg+msg

    def input(self):
        premsg=Colors.start + Colors.blue_CL  + "[INPUT]   ::" + Colors.end + " "
        return premsg

    def newline(self,verbose=None):
        return "\n             "

    def indent(self,verbose=None):
        return "             "
