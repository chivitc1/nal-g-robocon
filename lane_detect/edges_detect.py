import cv2
import numpy as np
import opencv_utils


def realtime_detect_edges():
    # capture frames from a camera
    cap = cv2.VideoCapture(0)

    # loop runs if capturing has been initialized
    while (1):

        # reads frames from a camera
        ret, frame = cap.read()

        # converting BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of red color in HSV
        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        # create a red HSV colour boundary and
        # threshold HSV image
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Display an original image
        cv2.imshow('Original', frame)

        # finds edges in the input image image and
        # marks them in the output map edges
        edges = cv2.Canny(frame, 100, 200)

        # Display edges in a frame
        cv2.imshow('Edges', edges)

        # Wait for Esc key to stop
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    # Close the window
    cap.release()

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


def image_edge_detect():
    img = cv2.imread('../images/2.jpg')

    roi = opencv_utils.get_roi_points(img, roi_heigh=130, roi_width_offset=50)
    roi_img = opencv_utils.get_roi_img(img, roi)
    cv2.imshow('Crop ROI', roi_img)

    # Convert BGR to HSV
    hsv_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('HSV img', hsv_img)

    # Masking image with color range
    # define range of black color in HSV
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([180, 255, 87])

    # create a red HSV colour boundary and threshold HSV image
    mask = cv2.inRange(hsv_img, lower_color, upper_color)

    # Bitwise-AND mask and original image
    masked_img = cv2.bitwise_and(roi_img, roi_img, mask=mask)

    cv2.imshow('Masked img', masked_img)

    # Blur image
    image_blured = cv2.bilateralFilter(masked_img, 13, 75, 75)
    # cv2.imshow('Bilateral filter Blured', image_blured)

    low_threshold = 50
    high_threshold = 150
    edges_img = cv2.Canny(image_blured, low_threshold, high_threshold)

    cv2.imshow('Edges', edges_img)

    kernel = np.ones((5, 5), np.uint8)
    # erosion_img = cv2.erode(masked_img, kernel, iterations=1)
    # cv2.imshow('Erosion img', erosion_img)

    dilation_img = cv2.dilate(masked_img, kernel, iterations=1)
    # cv2.imshow('dilation_img img', dilation_img)

    opening_img = cv2.morphologyEx(masked_img, cv2.MORPH_OPEN, kernel)
    cv2.imshow('opening_img img', opening_img)

    _, cnts, _ = cv2.findContours(edges_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over our contours
    screenCnt = None
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.3 * peri, True)

        cv2.drawContours(masked_img, [cnts[0]], -1, (0, 255, 0), 2)

        # if our approximated contour has four points, then
        # we can assume that we have found our card
        if len(approx) == 4:
            screenCnt = approx;
        break
    cv2.imshow('Draw contour', edges_img)

    hough_lines_img = get_houghs(edges_img)
    cv2.imshow('Hough lines img', hough_lines_img)

    # Free display windows when press a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_houghs(masked_edges):
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 20  # angular resolution in radians of the Hough grid
    threshold = 60  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 30  # maximum gap in pixels between connectable line segments
    line_image = np.copy(masked_edges) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    line_width=3
    line_color=(0,0,255)

    if lines is None:
        return masked_edges
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    lines_edges = cv2.addWeighted(masked_edges, 0.8, line_image, 1, 0)
    return lines_edges


if __name__ == '__main__':
    # realtime_detect_edges()
    image_edge_detect()