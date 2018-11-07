# Python program for Detection of a
# specific color(blue here) using OpenCV with Python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def process_img():
    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Black color
        lower_hsv_color = np.array([0, 17, 3])
        upper_hsv_color = np.array([187, 63, 87])

        # Red color
        # lower_hsv_color = np.array([0, 100, 100])
        # upper_hsv_color = np.array([20, 255, 255])

        # # Blue
        # lower_hsv_color = np.array([110, 50, 50])
        # upper_hsv_color = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower_hsv_color, upper_hsv_color)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('hello', res)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Destroys all of the HighGUI windows.
    cv2.destroyAllWindows()

    # release the captured frame
    cap.release()


if __name__ == '__main__':
    process_img()