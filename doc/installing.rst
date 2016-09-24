.. highlight:: rest

***************
Installing ALIS
***************

This document will describe how to install ALIS.

Installing Dependencies
=======================
ALIS uses a few python packages that are generally
considered to be commonly used in the scientific
community. The aim is to keep these required
packages to an absolute minimum.

In general, it is recommended that you use Anaconda for these
installations.

Detailed installation instructions are presented below:

Python Dependencies
-------------------

ALIS depends on the following list of Python packages.

It is recommended that you use `Anaconda <https://www.continuum.io/downloads/>`_ to install and/or update these packages.

* `python <http://www.python.org/>`_ versions 2.7, or 3.3 or later
* `numpy <http://www.numpy.org/>`_ version 1.10 or later
* `astropy <http://www.astropy.org/>`_ version 1.1 or later
* `scipy <http://www.scipy.org/>`_ version 0.17 or later
* `matplotlib <http://matplotlib.org/>`_  version 1.4 or later

If you are using Anaconda, you can check the presence of these packages with::

	conda list "python|numpy|astropy|scipy|matplotlib"

If the packages have been installed, this command should print out all the packages and their version numbers.

If any of these packages are missing you can install them with a command like::

	conda install astropy

If any of the packages are out of date, they can be updated with a command like::

	conda update scipy

Installing ALIS
===============

It is recommended that you grab the ALIS code from github::

	#go to the directory where you would like to install ALIS.
	git clone https://github.com/rcooke-ast/ALIS.git

From there, you can build and install either with install or develop, e.g.::

	cd ALIS
	python setup.py develop

or::

	cd ALIS
	python setup.py install

This should compile all the necessary files, etc.

Tests
=====
In order to assess whether ALIS has been properly installed,
we suggest you run the following tests:

1. Ensure run_alis works
------------------------
Go to a directory outside of the ALIS directory (e.g. your home directory),
then type run_alis.::

	cd
	run_alis

2. Run the ALIS unit tests
--------------------------

Enter the ALIS directory and do::

	python setup.py test

3. Try the test suite
---------------------
A few simple examples are provided in the directory ALIS/examples/

then follow the README file.
