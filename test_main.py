import pytest
from detect import *
from utils import *

curDegree = 0
def calCurrentDegree():
	global curDegree
	return curDegree

countDetector = RotateCountDetector()
def shouldChangeRotate(param):
	global countDetector
	countDetector.feedCurrent(calCurrentDegree())
	if countDetector.getRotatedCount() >= param:
		countDetector = RotateCountDetector()
		return True
	else:
		return False

def testShouldChangeRotate1():
	global curDegree
	curDegree = 0
	assert shouldChangeRotate(1) == False
	curDegree = 180
	assert shouldChangeRotate(1) == False
	curDegree = 0
	assert shouldChangeRotate(1) == True

def testShouldChangeRotate2():
	global curDegree
	curDegree = 340
	assert shouldChangeRotate(1) == False
	curDegree = 355
	assert shouldChangeRotate(1) == False
	curDegree = 175
	assert shouldChangeRotate(1) == False
	curDegree = 5
	assert shouldChangeRotate(1) == True
