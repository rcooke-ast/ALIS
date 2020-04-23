import numpy as np
import inspect
import sys

class dlas:
	def __init__(self, name):
		self._path="/path/to/my/project/fitting/"
		try:
			return eval("self."+name+"()")
		except:
			print "No such DLA! Try one of these:\n"
			members = [x for x, y in inspect.getmembers(self, predicate=inspect.ismethod)]
			for m in members:
				print m

	def object1(self):
		self._zabs = 2.0
		self._path += "object1/data/"
		self._filename = "object1.dat"

	def object2(self):
		self._zabs = 3.0
		self._path += "object2/data/"
		self._filename = "object2.dat"

	def object3(self):
		self._zabs = 4.0
		self._path += "object3/data/"
		self._filename = "object3.dat"
