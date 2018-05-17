from detect import *
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
	{"action": {"type": "rotate", "param": "left"}, "stop": {"type": "count", "param": 2}},

	{"action": {"type": "forward", "param": 0}, "stop": {"type": "distance", "param": 15}},
]

def doActionWait():
	stop()

def doActionForward(degree):
	rotateNearest(degree)
	forward()

totalCount = 0
prev = 0
def doActionRotate(direction):
	rotate(direction)

def doAction(action, param):
	if action == "wait":
		doActionWait()
	elif action == "forward"
		doActionForward(param)
	elif action == "rotate":
		doActionRotate(param)

def shouldChangeColor(param):
	if detectColor(100) == param:
		return True
	else
		return False

def shouldChangeObstacle(param):
	if detectObstacle(100, param):
		return True
	else
		return False

def shouldChangeRotate(param):
	curDegree = calCurrentDegree()
	delta = calSmallestDegree(curDegree, prev)
	# FIXME: calculate correct
	if delta < 5:
		totalCount = totalCount + 1
		prev = calSmallestDegree(180, prev)

	if totalCount >= param:
		totalCount = 0
		prev = 0
		return True
	else
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
	configureSpeed(30)
	test(90, 20)