from time import sleep
from action import Action
from basic import *
from detect import *
# import csv
# import time
# from io import StringIO


def run():
    action = Action()
    rgb = getColor()
    color = detect_color(rgb)
    if color == 'blue':
        action.go_straight(7)

    sleep(2)
    action.go_left(10)
    sleep(2)
    action.go_straight(5)
    sleep(2)
    action.go_left(10)

    # turnLeft()
    # sleep(0.2)
    # pause()
    # action.process_img()

    # with open(file='black.csv', mode='a') as resultFile:
    #     wr = csv.writer(resultFile, dialect='excel')
    #     for i in range(300):
    #         color = getColor()
    #         row = [color['red'], color['green'], color['blue'], 'black']
    #         wr.writerow(row)
    #         print(i)
    #         sleep(0.5)
    #     resultFile.close()
    # changeSpeed(250)
    # goForward()
    # sleep(3)
    # stop()




if __name__ == "__main__":
    # command = sys.argv
    # configureSpeed(30)
    run()
