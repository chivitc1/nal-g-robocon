from detect import *

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

def run():
	degree = 270
	while True:
		rotateNearest(degree)
		stepForward()
		wait(0.01)
	pause()

if __name__ == "__main__":
	command = sys.argv
	configureSpeed(30)
	run()