# Once you have created your own working function to be
# implemented in ALIS, you need to import it here. For
# example, if you create a function called 'myfunction.py'
# simply import it as follows:
#
# import myfunction

##########################
#  IMPORT FUNCTIONS HERE #

##########################

def load_user_functions():
    # If your function requires the atomic data used by ALIS,
    # you need to include the name of your function in the
    # sendatomic list: e.g. sendatomic = ['myfunc'] where
    # 'myfunc' is given by the parameter called self._idstr
    # in your function.
    sendatomic = []

    # Finally, add your new function to the following dictionary.
    # The key is simply the name you give to the parameter
    # self._idstr in your function, and the keyword arguments
    # are a call to your function Class. For example:
    # usrdict = dict({ 'myfunc'         : myfunction.MyFunc,
    #                  'another'        : another_function.Another
    #                 })
    usrdict = dict({})

    return usrdict, sendatomic
