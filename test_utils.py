from utils import *

def testReturnTrueWhenRotateGreaterThan10Degree():
	hangDetector = RotateHangDetector(100)
	hangDetector.feedCurrent(101)
	hangDetector.feedCurrent(110)
	assert (hangDetector.isHangOn() == False)
	assert (hangDetector.isRotated() == True)

def testReturnFalseWhenCanNotRotateAndEncounterLeastThan10Times():
	hangDetector = RotateHangDetector(100)
	hangDetector.feedCurrent(101)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	assert hangDetector.isHangOn() == False
	hangDetector.feedCurrent(101)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	assert hangDetector.isHangOn() == False
	assert hangDetector.isRotated() == False

def testReturnTrueWhenCanNotRotateAndEncounter10Times():
	hangDetector = RotateHangDetector(100)
	hangDetector.feedCurrent(101)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	assert hangDetector.isHangOn() == False
	hangDetector.feedCurrent(101)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	hangDetector.feedCurrent(103)
	assert hangDetector.isHangOn() == True
	assert hangDetector.isRotated() == False

def testReturn0WhenNotRotate():
	countDetector = RotateCountDetector()
	countDetector.feedCurrent(101)
	assert countDetector.getRotatedCount() == 0
	countDetector.feedCurrent(103)
	countDetector.feedCurrent(103)
	assert countDetector.getRotatedCount() == 0

def testReturn0WhenRotate90():
	countDetector = RotateCountDetector()
	countDetector.feedCurrent(350)
	assert countDetector.getRotatedCount() == 0
	countDetector.feedCurrent(355)
	countDetector.feedCurrent(360)
	countDetector.feedCurrent(10)
	countDetector.feedCurrent(30)
	countDetector.feedCurrent(60)
	countDetector.feedCurrent(80)
	countDetector.feedCurrent(90)
	assert countDetector.getRotatedCount() == 0

def testReturn0WhenRotate180():
	countDetector = RotateCountDetector()
	countDetector.feedCurrent(350)
	assert countDetector.getRotatedCount() == 0
	countDetector.feedCurrent(355)
	countDetector.feedCurrent(360)
	countDetector.feedCurrent(10)
	countDetector.feedCurrent(80)
	countDetector.feedCurrent(170)
	assert countDetector.getRotatedCount() == 0

def testReturn1WhenRotate350():
	countDetector = RotateCountDetector()
	countDetector.feedCurrent(350)
	assert countDetector.getRotatedCount() == 0
	countDetector.feedCurrent(355)
	countDetector.feedCurrent(360)
	countDetector.feedCurrent(10)
	countDetector.feedCurrent(80)
	countDetector.feedCurrent(170)
	countDetector.feedCurrent(190)
	countDetector.feedCurrent(350)
	assert countDetector.getRotatedCount() == 1

def testReturn2WhenRotate350():
	countDetector = RotateCountDetector()
	countDetector.feedCurrent(10)
	countDetector.feedCurrent(90)
	countDetector.feedCurrent(190)
	countDetector.feedCurrent(270)
	countDetector.feedCurrent(350)
	countDetector.feedCurrent(10)
	countDetector.feedCurrent(90)
	countDetector.feedCurrent(190)
	countDetector.feedCurrent(270)
	countDetector.feedCurrent(350)
	assert countDetector.getRotatedCount() == 2

def testReturn2WhenRotate350():
	countDetector = RotateCountDetector()
	countDetector.feedCurrent(270)
	countDetector.feedCurrent(360)
	countDetector.feedCurrent(90)
	countDetector.feedCurrent(180)
	countDetector.feedCurrent(270)
	countDetector.feedCurrent(350)
	countDetector.feedCurrent(10)
	countDetector.feedCurrent(90)
	countDetector.feedCurrent(190)
	countDetector.feedCurrent(270)
	countDetector.feedCurrent(350)
	assert countDetector.getRotatedCount() == 2

def testReturn1When1stRotateRight():
	countDetector = RotateCountDetector()
	countDetector.feedCurrent(360)
	countDetector.feedCurrent(330)
	countDetector.feedCurrent(270)
	countDetector.feedCurrent(260)
	countDetector.feedCurrent(200)
	countDetector.feedCurrent(180)
	countDetector.feedCurrent(170)
	countDetector.feedCurrent(90)
	countDetector.feedCurrent(80)
	countDetector.feedCurrent(10)
	countDetector.feedCurrent(0)
	assert countDetector.getRotatedCount() == 1

def testReturn2When2ndRotateRight():
	countDetector = RotateCountDetector()
	countDetector.feedCurrent(360)
	countDetector.feedCurrent(330)
	countDetector.feedCurrent(270)
	countDetector.feedCurrent(260)
	countDetector.feedCurrent(200)
	countDetector.feedCurrent(180)
	countDetector.feedCurrent(170)
	countDetector.feedCurrent(90)
	countDetector.feedCurrent(80)
	countDetector.feedCurrent(10)
	countDetector.feedCurrent(0)

	countDetector.feedCurrent(360)
	countDetector.feedCurrent(330)
	countDetector.feedCurrent(270)
	countDetector.feedCurrent(260)
	countDetector.feedCurrent(200)
	countDetector.feedCurrent(180)
	countDetector.feedCurrent(170)
	countDetector.feedCurrent(90)
	countDetector.feedCurrent(80)
	countDetector.feedCurrent(10)
	countDetector.feedCurrent(0)
	assert countDetector.getRotatedCount() == 2