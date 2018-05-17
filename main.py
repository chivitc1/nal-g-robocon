from detect import *
from utils import *
from collections import deque

states = [
	{"action": {"step": "wait"}, "stop": {"color": "green"}},
	{"action": {"step": "forward", "degree": 270}, "stop": {"distance": 15}},
	{"action": {"step": "forward", "degree": 0}, "stop": {"distance": 15}},
	{"action": {"step": "rotate", "count": 2}, "stop": {"degree": 270}},

	{"action": {"step": "forward", "degree": 0}, "stop": {"distance": 15}},


	{"action": {"type": "forward", "degree": 90}, "stop": {"distance": 15}},
	{"action": {"type": "rotate", "count": 2}, "stop": {"degree": 180}},

	{"action": {"type": "forward", "degree": 90}, "stop": {"distance": 15}},
	{"action": {"type": "rotate", "count": 2}, "stop": {"degree": 180}},

	{"action": {"type": "forward", "degree": 180}, "stop": {"distance": 15}},
	{"action": {"type": "forward", "degree": 270}, "stop": {"distance": 15}},
	{"action": {"type": "forward", "degree": 180}, "stop": {"distance": 15}},


	{"action": {"type": "forward", "degree": 270}, "stop": {"distance": 15}},
	{"action": {"type": "forward", "degree": 0}, "stop": {"distance": 15}},
	{"action": {"type": "rotate", "count": 2}, "stop": {"degree": 270}},

	{"action": {"type": "forward", "degree": 0}, "stop": {"distance": 15}},
]

states = [
	{"action": {"type": "wait", "param": 0}, "stop": {"type": "color", "param": "green"}},
	{"action": {"type": "forward", "param": 270}, "stop": {"type": "distance", "param": 15}},
	{"action": {"type": "forward", "param": 0}, "stop": {"type": "distance", "param": 15}},
	{"action": {"type": "rotate", "param": "right"}, "stop": {"type": "count", "param": 2}},
	{"action": {"type": "forward", "param": 270}, "stop": {"type": "distance", "param": 15}},

	{"action": {"type": "forward", "param": 0}, "stop": {"type": "distance", "param": 15}},
	{"action": {"type": "stop", "param": 0}, "stop": {"type": "color", "param": "blue"}},
]

def doActionWait():
	stop()

def doActionForward(degree):
	rotateNearest(degree)
	forward()

def doActionRotate(direction):
	rotate(direction)

def doActionStop():
	pass

def doAction(action, param):
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
		return True
	else:
		return False

def shouldChangeObstacle(param):
	if detectObstacle(100, param):
		return True
	else:
		return False

def shouldStop(param):
	if detectColor(500) == param:
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
	elif cond == "stop":
		return shouldStop(param)
	return False

def run():
	stateQueue = deque(states)
	state = stateQueue.popLeft()
	while len(stateQueue) > 0:
		doAction(state["action"]["type"], state["action"]["param"])
		wait(10)
		shouldChange = shouldChangeState(state["stop"]["type"], state["stop"]["param"])
		if shouldChange:
			stop()
			state = stateQueue.popLeft()
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
	# configureSpeed(30)
	# test(90, 20)
	# 
	while True:
		detectObstacle(100, 15)
