from basic import *
import sys
import math

def detectColor():
	colors = getColor()
	red = colors["red"]
	green = colors["green"]
	blue = colors["blue"]
	print(str(colors) + "\n")
	color = "undetected"
	if blue > 4800 and (red + green) < 10000:
		color = "blue"
	if red > 4800 and (blue + green) < 10000:
		color = "red"
	if green > 4800 and (blue + red) < 10000:
		color = "green"
	if green > 10000 and blue > 10000 and red > 10000:
		color = "white"
	return color

def detectColor2():
	while True:
		print("detect color is: " + detectColor() + "\n")
		wait(1)

def stepRotate(isLeft):
	if isLeft:
		print("Rotate left\n")
		turnLeft()
	else:
		print("Rotate right\n")
		turnRight()

def calCurrentDegree():
	currentDegree = getCompass()["degree"]
	# baseDegree = 270
	# degree = currentDegree + 90
	# if degree > 360:
	# 	degree = degree - 360
	# return degree
	
	# return currentDegree
	
	if currentDegree > 250:
		currentDegree = currentDegree + 20
	elif currentDegree > 160:
		currentDegree = currentDegree + 10

	return currentDegree
	

def calDelta(a, b):
	return (a - b) % 360

def checkShouldTurn(degree, dd): 
	currentDegree = calCurrentDegree()
	if calDelta(degree, currentDegree) > dd:
		return "left"
	elif calDelta(degree, currentDegree) < - dd:
		return "right"
	else:
		return None

def rotate(isLeft, degree):
	# changeSpeed(10)
	delta = 5
	while True:
		currentDegree = calCurrentDegree()
		if calDelta(degree, currentDegree) > delta:
			stepRotate(True)
		elif calDelta(degree, currentDegree) < - delta:
			stepRotate(False)
		else:
			break
		wait(0.09)

def rotateNearest(degree):
	# changeSpeed(10)
	delta = 10
	lastDegree = 0
	hangCount = 0
	while True:
		currentDegree = calCurrentDegree()
		if degree > currentDegree:
			isLeft = True
		else:
			isLeft = False
		leftDelta = math.fabs(degree - currentDegree)
		print("leftDelta " + str(leftDelta))
		if leftDelta > 200:
			isLeft = not isLeft
		print("at " + str(currentDegree) + "should go left(" + str(isLeft) + ") to reach " + str(degree))

		if leftDelta > delta:
			stepRotate(isLeft)
			if currentDegree - lastDegree < 5:
				hangCount = hangCount + 1
			if hangCount > 10:
				print("Is hang")
				wait(0.5)
				# resetCompass()
				hangCount = 0
			else:
				wait(0.1)
			lastDegree = currentDegree
			pause()
		else:
			break

def checkChangeState():
	distance = getDistance()
	if distance < 30:
		return True
	else:
		return False

def stepGo():
	print("Go\n")
	goForward()

def run():
	while True:
		rotateNearest(degree)
		stepGo()
		wait(0.01)
	pause()

states = [
	{"action": {"step": "forward", "degree": 270}, "stop": {"distance": 15}},
	{"action": {"step": "forward", "degree": 0}, "stop": {"distance": 15}},
	{"action": {"step": "rotate", "count": 2}, "stop": {"degree": 270}},
	{"action": {"step": "forward", "degree": 0}, "stop": {"time": 2}},
	{"action": {"step": "forward", "degree": 90}, "stop": {"distance": 15}},
	{"action": {"step": "forward", "degree": 0}, "stop": {"distance": 45}},
	{"action": {"step": "rotate", "count": 2}, "stop": {"degree": 0}},
	{"action": {"step": "backward", "degree": 0}, "stop": {"distance": 70}},
	{"action": {"step": "rotate", "count": 1}, "stop": {"degree": 270}},
	{"action": {"step": "rotate", "count": 1}, "stop": {"degree": 270}},
]

states = [
	{"action": {"step": "forward", "degree": 270}, "stop": {"distance": 15}},
	{"action": {"step": "forward", "degree": 0}, "stop": {"distance": 15}},
	{"action": {"step": "rotate", "count": 2}, "stop": {"degree": 270}},
	{"action": {"step": "forward", "degree": 0}, "stop": {"time": 2}},
	{"action": {"step": "forward", "degree": 90}, "stop": {"distance": 15}},
	{"action": {"step": "forward", "degree": 0}, "stop": {"distance": 45}},
	{"action": {"step": "rotate", "count": 2}, "stop": {"degree": 0}},
	{"action": {"step": "backward", "degree": 0}, "stop": {"distance": 70}},
	{"action": {"step": "rotate", "count": 1}, "stop": {"degree": 270}},
	{"action": {"step": "rotate", "count": 1}, "stop": {"degree": 270}},
]

if __name__ == "__main__":
	command = sys.argv
	changeSpeed(30)
	# detectColor2()
	# rotate(True, 0)
	run()