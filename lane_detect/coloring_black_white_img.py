import cv2
import numpy as np
import opencv_utils


def image_edge_detect():
    img = cv2.imread('../images/8.jpg')

    roi = opencv_utils.get_roi_points(img, roi_heigh=130, roi_width_offset=50)
    roi_img = opencv_utils.get_roi_img(img, roi)
    cv2.imshow('Crop ROI', roi_img)

    # Convert BGR to HSV
    hsv_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('HSV img', hsv_img)

    hsv_channels = cv2.split(hsv_img)

    rows = roi_img.shape[0]
    cols = roi_img.shape[1]

    for i in range(0, rows):
        for j in range(0, cols):
            h = hsv_channels[0][i][j]

            if h > 90 and h < 130:
                hsv_channels[2][i][j] = 255
            else:
                hsv_channels[2][i][j] = 0

    cv2.imshow("show", hsv_channels[0])
    cv2.imshow("show2", hsv_channels[2])

    # Free display windows when press a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # realtime_detect_edges()
    image_edge_detect()