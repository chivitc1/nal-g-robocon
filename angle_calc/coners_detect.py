# Python program for Detection of a
# specific color(blue here) using OpenCV with Python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def process_img():
    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = get_transformed_img(hsv)
        cv2.imshow('hello', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


def get_transformed_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect corners with the goodFeaturesToTrack function.
    corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10)
    corners = np.int0(corners)

    # we iterate through each corner,
    # making a circle at each point that we think is a corner.
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
    return img


if __name__ == '__main__':
    process_img()