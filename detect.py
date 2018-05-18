from basic import *
from utils import *
from time import sleep
import sys
import math

"""Configure speed"""
def configureSpeed(speed):
	changeSpeed(speed)

def stop():
	pause()

def wait(ms):
	sleep(0.001 * ms)

"""Detect color in single step"""
def stepDetectColor():
	colors = getColor()
	red = colors["red"]
	green = colors["green"]
	blue = colors["blue"]
	# print("Step color: " + str(colors) + "\n")
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

"""Detect color within specific timebbound in second"""
def detectColor(time):
	total = {"red": 0, "green": 0, "blue": 0, "white": 0, "undetected": 0}
	count = time / 30
	for i in range(0, count):
		color = stepDetectColor()
		total[color] = total[color] + 1
		wait(30)
	color = "undetected"
	if total["green"] > total[color]:
		color = "green"
	if total["blue"] > total[color]:
		color = "blue"
	if total["red"] > total[color]:
		color = "red"
	print("Detect color " + color + " with count " + str(total[color]) + "\n")
	return color

"""Rotate in single step"""
def stepRotate(isLeft):
	if isLeft:
		# print("Step rotate: left\n")
		turnLeft()
	else:
		# print("Step rotate: right\n")
		turnRight()

def rotate(isLeft):
	hangDetector = RotateHangDetector(calCurrentDegree())
	while not hangDetector.isRotated():
		if isLeft:
			turnLeft()
		else:
			turnRight()
		if hangDetector.isHangOn():
			wait(500)
		else:
			wait(300)
		stop()
		hangDetector.feedCurrent(calCurrentDegree())

"""Calculate current degree, resolve difference"""
def calCurrentDegree():
	currentDegree = getCompass()["degree"]
	# baseDegree = 270
	# degree = currentDegree + 90
	# if degree > 360:
	# 	degree = degree - 360
	# return degree
	
	# return currentDegree
	
	if currentDegree < 50:
		currentDegree = currentDegree - 10
	elif currentDegree > 250:
		currentDegree = currentDegree + 10
	elif currentDegree > 160:
		currentDegree = currentDegree + 10

	currentDegree = currentDegree + 90
	if currentDegree >= 360:
		currentDegree = currentDegree - 360

	return currentDegree

def detectDegree(ms):
	resetCompass()
	count = ms / 30
	totalDegree = 0
	for i in range(0, count):
		totalDegree += getCompass()["degree"]
		wait(30)
	currentDegree = totalDegree / count
	# print("Current degree " + str(currentDegree) + "\n")
	return currentDegree


"""Calculate different between 2 degrees, return smallest angle(<180)"""
def calSmallestDegree(degree1, degree2):
	delta = math.fabs(degree1 - degree2)
	if delta >= 180:
		delta = delta - 180
	return delta

"""Try rotating base on smallest right"""
def rotateNearest(degree):
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
		# print("leftDelta " + str(leftDelta) + "\n")
		if leftDelta > 200:
			isLeft = not isLeft
		# print("at " + str(currentDegree) + "should go left(" + str(isLeft) + ") to reach " + str(degree) + "\n")

		if leftDelta > delta:
			stepRotate(isLeft)
			if currentDegree - lastDegree < 5:
				hangCount = hangCount + 1
			if hangCount > 10:
				# print("Is hang\n")
				wait(500)
				# resetCompass()
				hangCount = 0
			else:
				wait(100)
			lastDegree = currentDegree
			pause()
		else:
			break

def stepDetectObstacle(length):
	distance = getDistance()
	if distance < length:
		return True
	else:
		return False

def detectObstacle(time, length):
	total = 0
	count = time / 50
	for i in range(0, count):
		if stepDetectObstacle(length):
			total = total + 1
		wait(50)
	hasObstacle = total > count/2
	print("Does encounter obstacle: " + str(hasObstacle) + "\n")
	return hasObstacle

def keepDistance(length):
	shouldPause = stepDetectObstacle(length)
	if shouldPause:
		pause()
	return shouldPause

def forward():
	print("Step go: forward\n")
	goForward()

def backward():
	print("Step go: backward\n")
	goBackward()