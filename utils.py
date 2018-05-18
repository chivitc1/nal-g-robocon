import math

class RotateHangDetector(object):

	def __init__(self, currentDegree):
		self.originDegree = currentDegree
		self.isHang = False
		self.isRot = False
		self.count = 0

	def feedCurrent(self, currentDegree):
		if math.fabs(self.originDegree - currentDegree) < 10:
			self.count = self.count + 1
			if self.count >= 10:
				self.isHang = True
		else:
			self.isRot = True
			self.isHang = False

	def isRotated(self):
		return self.isRot

	def isHangOn(self):
		return self.isHang


"""Calculate different between 2 degrees, return smallest angle(<180)"""
def calSmallestDegree(degree1, degree2):
	delta = math.fabs(degree1 - degree2)
	return delta

class RotateCountDetector(object):

	def __init__(self):
		self.count = 0
		self.pos = "first"

	def feedCurrent(self, currentDegree):
		delta1 = calSmallestDegree(currentDegree, 0)
		delta2 = calSmallestDegree(currentDegree, 360)
		if delta1 <= 30 or delta2 <= 30:
			if self.pos != "first":
				self.count = self.count + 1
				self.pos = "first"

		delta = calSmallestDegree(currentDegree, 180)
		# print("delta" + str(delta))
		if delta <= 30:
			if self.pos != "second":
				self.pos = "second"

	def getRotatedCount(self):
		return self.count