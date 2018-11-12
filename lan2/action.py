from basic import *
from time import sleep

import matplotlib.image as mpimg

import numpy as np
import cv2


class Action:

    PAUSE = 0.05
    LEFT_PAUSE = 0.05

    def go_straight(self, distance):
        while True:
            goForward()
            sleep(self.PAUSE)
            pause()
            detect_obstackle = getDistance()
            print(detect_obstackle)
            if 0 < detect_obstackle <= distance:
                break

    def go_left(self, max):
        count = 0
        while count < max:
            turnLeft()
            sleep(self.LEFT_PAUSE)
            pause()
            count = count + 1

    def process_img(self):
        # reading in an image
        # image = mpimg.imread('img/3.jpg')
        data_bytes = bytes()
        stream = get_image()
        while True:
            data_bytes += stream.read(1024)
            a = data_bytes.find(b'\xff\xd8')
            b = data_bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = data_bytes[a:b + 2]
                img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                break
        cv2.imwrite('test.jpg', img)

        image = img

        # image = mpimg.imread('img/img_a2.jpg')

        height = image.shape[0]
        width = image.shape[1]

        # Convert to grayscale here.
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cannyed_image = cv2.Canny(gray_image, 200, 300)

        region_of_interest_vertices = [
            (0, height),
            (width / 2, height / 2),
            (width, height),
        ]

        # cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32),)

        # Detect line
        lines = cv2.HoughLinesP(
            cannyed_image,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25
        )

        for line in lines:
            for x1, y1, x2, y2 in line:
                # if (y1==y2 or abs(int(x1)-int(x2))/abs(int(y1)-int(y2)) < 0.2):
                if (int(x1) > 400 or int(x2) > 400):
                    if (int(y1) > 300 or int(y2) > 300):
                        print("trai")
                    else:
                        if (int(y1) < 200 or int(y2) < 200):
                            print("phai")
                        else:
                            print("đung ")
                    return




