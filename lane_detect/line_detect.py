import glob
import pandas as pd
import cv2
import numpy as np
import opencv_utils

DEGREE_UNIT = 10    # 5 degree as one unit
MIN_SLOPE, MAX_SLOPE = - DEGREE_UNIT * np.pi/180, DEGREE_UNIT * np.pi/180

DETECT_LINE_ANGLE_MAX = 80

cap = cv2.VideoCapture(0)


def realtime_lane_detect():
    count = 0
    while True:
        _, img = cap.read()
        count = count + 1
        if count == 3:
            count = 0
        if count == 1 or count == 2:
            cv2.imshow('Detected line and deviation', img)
            continue
        img2 = get_img_detected(img)
        cv2.imshow('Detect lane', img2)

        k = cv2.waitKey(2) & 0xFF
        if k == 27:
            break

        # Destroys all of the HighGUI windows.
    cv2.destroyAllWindows()

    # release the captured frame
    cap.release()



def test_image_lane_detect():
    """
    Load images from file
    draw single line of lane detect, deviation angle, distance to vertical center line of screen
    :return:
    """
    file= "/home/chinv/dev/projects/robocon/robo/images/1.jpg"
    # Process line detect
    img = cv2.imread(file)
    print(file)
    img2 = get_img_detected(img)
    cv2.imshow(file, img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_images_lane_detect():
    """
    Load images from file
    draw single line of lane detect, deviation angle, distance to vertical center line of screen
    :return:
    """
    files = read_images()
    for file in files:
        # Process line detect
        img = cv2.imread(file)
        print(file)
        img2 = get_img_detected(img)
        cv2.imshow(file, img2)

        cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_images():
    """
    Load test images from files
    :return:
    """
    files = glob.glob("../images/*.jpg")
    return files


def get_img_detected(img):
    roi = opencv_utils.get_roi_points(img, roi_heigh=150, roi_width_offset=10)
    roi_img = opencv_utils.get_roi_img(img, roi)
    cv2.imshow('Crop ROI', roi_img)

    # Convert BGR to HSV
    hsv_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV img', hsv_img)

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
    cv2.imshow('Bilateral filter Blured', image_blured)

    low_threshold = 50
    high_threshold = 150
    edges_img = cv2.Canny(image_blured, low_threshold, high_threshold)
    cv2.imshow('Edge by Canny', edges_img)

    lines = get_detected_line(edges_img)

    if not lines:
        return img
    if lines["left"]:
        line = lines["left"]
        cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), thickness=3)
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(img, "degree {}, deviation {}".format(line.degree, line.distance), (100, 50), font, 1.0,
                    (0, 0, 255), 1)

        print("deviation degree: {}".format(line.degree))
        print("deviation distance: {}".format(line.distance))

    if lines["right"]:
        line = lines["right"]
        cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), thickness=3)
        cv2.putText(img, "degree {}, deviation {}".format(line.degree, line.distance), (100, 50), font, 1.0,
                    (0, 0, 255), 1)

    return img


def get_detected_line(img):
    """
    :param img: edged img
    :return: detected line
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    print(img.shape)
    # This returns an array of r and theta values
    """
    Grid scale = rho
    Angle = theta
    Number of votes = threshold
    """

    houged_lines = cv2.HoughLines(img, rho=1, theta=np.pi / 20, threshold=30)
    if houged_lines is None:
        return None
    lines = []
    left_lines = []
    right_lines = []
    top_horizontal_line = opencv_utils.LineXY(0, 0, img_width, 0)
    bottom_horizontal_line = opencv_utils.LineXY(0, img_height, img_width, img_height)

    vertical_center_line = opencv_utils.LineXY(x1=int(img_width / 2), y1=0, x2=int(img_width / 2), y2=img_height)

    horizontal_center_line = opencv_utils.LineXY(x1=0, y1=int(img_height / 2), x2=img_width, y2=int(img_height /2))

    for line in houged_lines:
        for r, theta in line:

            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a * r

            # y0 stores the value rsin(theta)
            y0 = b * r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000 * (a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000 * (a))

            degree = theta * 180 / np.pi

            if degree > 90:
                degree = -180 + degree
            # round degree by 10
            degree = opencv_utils.round_angle(DEGREE_UNIT, degree)

            print("x1 {}, y1 {}, x2 {}, y2 {}, degree: {}".format(x1, y1, x2, y2, degree))
            if np.abs(degree) > DETECT_LINE_ANGLE_MAX:
                continue

            a_line = opencv_utils.LineXY(x1, y1, x2, y2)
            top_cross_point = a_line.intersect(top_horizontal_line)
            bottom_cross_point = a_line.intersect(bottom_horizontal_line)
            if not top_cross_point:
                continue
            # print("DEGREE DETECT: {}".format(degree))

            # print("x1 {}, y1 {}, x2 {} , y2 {} Cross point {}, {}. Degree {}".format(x1, y1, x2, y2, top_cross_point.x, top_cross_point.y, degree))
            if degree <= 0:
                left_lines.append({"slope": degree, "top_x": top_cross_point.x, "bottom_x": bottom_cross_point.x})
            else:
                right_lines.append({"slope": degree, "top_x": top_cross_point.x, "bottom_x": bottom_cross_point.x})
        if len(left_lines) == 0 and len(right_lines) ==0:
            return None


    # filter lines
    left_df_lines = pd.DataFrame({'slope': [item["slope"] for item in left_lines],
                             'top_x': [item["top_x"] for item in left_lines],
                             'bottom_x': [item["bottom_x"] for item in left_lines]})
    right_df_lines = pd.DataFrame({'slope': [item["slope"] for item in right_lines],
                                  'top_x': [item["top_x"] for item in right_lines],
                                  'bottom_x': [item["bottom_x"] for item in right_lines]})

    print("---------------------------------------")

    result = {}
    if len(left_lines) == 0:
        result["left"] = None

    else:
        left_slope_std = None
        if len(left_lines) == 1:
            print("left len {}".format(len(left_lines)))
            left_slope_std = left_lines[0]["slope"]
        else:
            left_slope_std = left_df_lines.slope.std()
        left_df_lines = left_df_lines[np.abs(left_df_lines.slope - left_df_lines.slope.mean()) <= (2 * left_slope_std)]

        x1 = int(left_df_lines.top_x.mean())
        y1 = 0
        x2 = int(left_df_lines.bottom_x.mean())
        y2 = img_height
        degree = int(left_df_lines.slope.mean())
        cross_point = opencv_utils.LineXY(x1, y1, x2, y2).intersect(horizontal_center_line)

        distance = 0
        if cross_point:
            distance = int(abs(vertical_center_line.x1 - cross_point.x))

        result["left"] = LaneDetected(degree, distance, x1, y1, x2, y2)

    if len(right_lines) == 0:
        result["right"] = None
    else:
        right_std_slope = None
        if len(right_lines) == 1:
            right_std_slope = right_lines[0]["slope"]
        else:
            std_slope = right_df_lines.slope.std()
        right_df_lines = right_df_lines[
            np.abs(right_df_lines.slope - right_df_lines.slope.mean()) <= (2 * right_std_slope)]

        right_x1 = int(right_df_lines.top_x.mean())
        right_y1 = 0
        right_x2 = int(right_df_lines.bottom_x.mean())
        right_y2 = img_height
        right_degree = int(right_df_lines.slope.mean())
        right_cross_point = opencv_utils.LineXY(right_x1, right_y1, right_x2, right_y2).intersect(
            horizontal_center_line)
        right_distance = 0
        if right_cross_point:
            right_distance = int(abs(vertical_center_line.x1 - right_cross_point.x))
        result["right"] = LaneDetected(right_degree, right_distance, right_x1, right_y1, right_x2, right_y2)

    # return {"degree": degree, "distance": distance, "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}

    return result


class LaneDetected:
    def __init__(self, degree, distance, x1, y1, x2, y2):
        self.degree = degree
        self.distance = distance
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


def radian_to_degree(slope):
    degree = slope * 180 / np.pi
    opencv_utils.round_angle(DEGREE_UNIT, degree)
    return degree


if __name__ == '__main__':
    # realtime_lane_detect()
    # draw_detect_lines()
    # test_images_lane_detect()
    test_image_lane_detect()