import requests
import json
import urllib

host = "http://10.22.40.101:8081"
host = "http://10.22.20.59:8080"
headers = {'Content-Type':'application/json'}

def turnLeft():
	url = host + "/remote/left"
	requests.get(url)

def turnRight():
	url = host + "/remote/right"
	requests.get(url)

def goForward():
	url = host + "/remote/play"
	requests.get(url)

def pause():
	url = host + "/remote/pause"
	requests.get(url)

def goBack():
	url = host + "/remote/back"
	requests.get(url)

def goBack():
	url = host + "/remote/back"
	requests.get(url)

def getColor():
	url = host + "/remote/color"
	res = requests.get(url)
	try:
		# print(res.text)
		return json.loads(res.text)
	except ValueError:
		print("Get color error\n")
		return {"red": 0, "green": 0, "blue": 0}

def getCompass():
	url = host + "/remote/compass"
	res = requests.get(url)
	return json.loads(res.text)

def resetCompass():
	url = host + "/remote/reset_compass"
	requests.get(url)

def getDistance():
	url = host + "/remote/sonar"
	res = requests.get(url)
	return json.loads(res.text)["distance"]

def changeSpeed(value):
	payload = {'speed': value}
	print(json.dumps(payload))
	url = host + "/remote/change_spd"
	requests.post(url, data = json.dumps(payload), headers = headers)

def changeSpeedRight(value):
	payload = {'speed': value}
	url = host + "/remote/change_spd_right"
	requests.post(url, data = json.dumps(payload), headers = headers)

def changeSpeedLeft(value):
	payload = {'speed': value}
	url = host + "/remote/change_spd_left"
	requests.post(url, data = json.dumps(payload), headers = headers)

def get_image():
	url = host + "/image_feed"
	stream = urllib.request.urlopen(url)
	return stream