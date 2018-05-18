from detect import *
from utils import *
from collections import deque
from random import *

"""forward, backward, left, right"""
class Direction(object):
	def __init__(self):
		self.direction = "forward"
		self.dirs = {
			"forward": {
				"left": "left",
				"right": "right",
				"degree": 270
			},
			"left": {
				"left": "backward",
				"right": "forward",
				"degree": 0
			},
			"backward": {
				"left": "right",
				"right": "left",
				"degree": 90
			},
			"right": {
				"left": "forward",
				"right": "backward",
				"degree": 180
			}
		}

	def turnLeft(self):
		self.direction = self.dirs[self.direction]["left"]

	def turnRight(self):
		self.direction = self.dirs[self.direction]["right"]

	def getDegree(self):
		return self.dirs[self.direction]["degree"]

direction = Direction()

def doActionGo():
	global direction
	rotateNearest(direction.getDegree())
	forward()

def doActionRotate(direction):
	if direction == "left":
		rotate(True)
	else:
		rotate(False)

def shouldChangeColor(param):
	print("Detect color " + param + "\n")
	if detectColor(100) == param:
		return True
	else:
		return False

def shouldChangeObstacle(param):
	if detectObstacle(100, param):
		return True
	else:
		return False

countDetector = RotateCountDetector()
def shouldChangeRotate(param):
	global countDetector
	countDetector.feedCurrent(calCurrentDegree())
	if countDetector.getRotatedCount() >= param:
		countDetector = RotateCountDetector()
		return True
	else:
		return False

def shouldStopWhenFoundColorOrRotate():
	if shouldChangeColor("red"):
		while True:
			doActionRotate("right")
			if shouldChangeRotate(2):
				break
	elif shouldChangeRotate("blue"):
		return True
	return False

def run():
	global direction
	ran = [True, False]
	while not shouldChangeColor("green"):
		wait(100)
	while True:
		resetCompass()
		if shouldChangeObstacle(23):
			shouldStop = shouldStopWhenFoundColorOrRotate()
			if shouldStop:
				stop()
				break
			shuffle(ran)
			if ran[0]:
				direction.turnLeft()
				if shouldChangeObstacle(23):
					direction.turnRight()
					direction.turnRight()
			else:
				direction.turnRight()
				if shouldChangeObstacle(23):
					direction.turnLeft()
					direction.turnLeft()
		else:
			doActionGo()
		wait(50)
	stop()

if __name__ == "__main__":
	command = sys.argv
	configureSpeed(25)
	run()
