import pytest
from detect import *

curDegree = 0
def calCurrentDegree():
	global curDegree
	return curDegree

totalCount = -1
pos = "first"

def shouldChangeRotate(param):
	global pos
	global totalCount
	curDegree = calCurrentDegree()
	delta = calSmallestDegree(curDegree, 0)
	if delta <= 5:
		if pos != "first":
			totalCount = totalCount + 1
			pos = "first"

	delta = calSmallestDegree(curDegree, 180)
	if delta <= 5:
		if pos != "second":
			pos = "second"

	print("Post " + pos + "\n")

	if totalCount >= param:
		totalCount = 0
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
