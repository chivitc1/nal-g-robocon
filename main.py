from detect import *
from utils import *
from collections import deque

maxStopDistance = 21

# open maps
states = [
	{"action": {"type": "wait", "param": 0}, "stop": {"type": "color", "param": "green"}},
	{"action": {"type": "forward", "param": 270}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "forward", "param": 0}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "rotate", "param": "right"}, "stop": {"type": "count", "param": 2}},
	{"action": {"type": "forward", "param": 270}, "stop": {"type": "distance", "param": maxStopDistance}},

	{"action": {"type": "forward", "param": 0}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "forward", "param": 90}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "forward", "param": 0}, "stop": {"type": "distance", "param": maxStopDistance}},

	{"action": {"type": "forward", "param": 90}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "rotate", "param": "right"}, "stop": {"type": "count", "param": 2}},

	{"action": {"type": "forward", "param": 270}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "stop", "param": 0}, "stop": {"type": "color", "param": "blue"}},
]

# hiden maps
states = [
	{"action": {"type": "wait", "param": 0}, "stop": {"type": "color", "param": "green"}},
	{"action": {"type": "forward", "param": 0}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "forward", "param": 270}, "stop": {"type": "distance", "param": maxStopDistance}},

	{"action": {"type": "rotate", "param": "right"}, "stop": {"type": "count", "param": 2}},

	{"action": {"type": "forward", "param": 0}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "forward", "param": 90}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "forward", "param": 0}, "stop": {"type": "distance", "param": 18}},

	{"action": {"type": "rotate", "param": "right"}, "stop": {"type": "count", "param": 2}},
	
	{"action": {"type": "forward", "param": 270}, "stop": {"type": "distance", "param": maxStopDistance}},

	{"action": {"type": "rotate", "param": "right"}, "stop": {"type": "count", "param": 2}},
	
	{"action": {"type": "forward", "param": 180}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "forward", "param": 270}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "forward", "param": 0}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "forward", "param": 270}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "forward", "param": 180}, "stop": {"type": "distance", "param": maxStopDistance}},
	{"action": {"type": "stop", "param": 0}, "stop": {"type": "color", "param": "blue"}},

]

def doActionWait():
	stop()

def doActionForward(degree):
	rotateNearest(degree)
	forward()

def doActionRotate(direction):
	if direction == "left":
		rotate(True)
	else:
		rotate(False)

def doActionStop():
	pass

def doAction(action, param):
	print("Do action " + action + " with param " + str(param) + "\n")
	if action == "wait":
		doActionWait()
	elif action == "forward":
		doActionForward(param)
	elif action == "rotate":
		doActionRotate(param)
	elif action == "stop":
		doActionStop()

def shouldChangeColor(param):
	if detectColor(100) == param:
		print("Detect color " + param + "\n")
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

def shouldChangeState(cond, param):
	if cond == "color":
		return shouldChangeColor(param)
	elif cond == "distance":
		return shouldChangeObstacle(param)
	elif cond == "count":
		return shouldChangeRotate(param)
	return False

def run():
	stateQueue = deque(states)
	state = stateQueue.popleft()
	# shouldChange = True
	while len(stateQueue) > 0:
		resetCompass()
		doAction(state["action"]["type"], state["action"]["param"])
		wait(10)
		shouldChange = shouldChangeState(state["stop"]["type"], state["stop"]["param"])
		if shouldChange:
			stop()
			state = stateQueue.popleft()
	stop()

def test(degree, length):
	while True:
		rotateNearest(degree)
		shouldPause = keepDistance(length)
		if not shouldPause:
			forward()
		wait(10)
	stop()

if __name__ == "__main__":
	command = sys.argv
	configureSpeed(30)
	run()

	# test(270, 20)


	# while True:
	# 	detectObstacle(100, 15)
