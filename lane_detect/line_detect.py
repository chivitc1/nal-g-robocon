import cv2
import numpy as np
import opencv_utils

# img = cv2.imread('../images/1.jpg')
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
        img2, detected_lines = get_detect_lines_img(img)
        cv2.imshow('Detect lane', img2)

        k = cv2.waitKey(2) & 0xFF
        if k == 27:
            break

        # Destroys all of the HighGUI windows.
    cv2.destroyAllWindows()

    # release the captured frame
    cap.release()


def get_detected_lines_and_degrees(img):
    """
    :param img: edged img
    :return: detected_lines in LineXY
    """
    DETECT_LINE_ANGLE_LIMIT=60
    DETECT_LINE_ANGLE_UNIT=5

    img_height = img.shape[0]
    img_width = img.shape[1]
    # This returns an array of r and theta values
    lines = cv2.HoughLines(img, rho=1, theta=np.pi / 20, threshold=30)
    if lines is None:
        return []
    # print("Num of lines {}".format(lines[0].size))
    degrees = []
    detected_lines = []
    deviation_distances = []

    for line in lines:
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
            degree = int(degree)
            if degree > 90:
                degree = -180 + degree
            # print("Degree {}".format(degree))

            # if np.abs(degree) > DETECT_LINE_ANGLE_LIMIT:
            #     continue
            # degree = opencv_utils.round_angle(DETECT_LINE_ANGLE_UNIT, degree)

            d = get_deviation_distance(img, x1, y1, x2, y2)
            if d != -1:
                deviation_distances.append(d)
                a_line = LineXY(x1, y1, x2, y2, degree, deviation_distance=d)
                detected_lines.append(a_line)
                degrees.append(degree)
                # print("Line: (x1, y1) {}, (x2, y2) {}, r {}, degree {}, deviation distance {}".format((x1, y1), (x2, y2), r, degree, d))
    filtered_lines = _filter_line_outliers(detected_lines, DETECT_LINE_ANGLE_LIMIT, img_width, img_height)
    return filtered_lines

    # Process when there are two lines with the same degree and distance deviation threshold
    # if len(detected_lines) == 2:
    #     print("merge two lines into one")
    #     pt1, pt2 = np.array([0, 0]), np.array([img_width, 0])
    #     pt3, pt4 = np.array([0, img_height]), np.array([img_width, img_height])
    #     horizontal_upper_line = opencv_utils.to_factored_line(pt1, pt2)
    #     horizontal_lower_line = opencv_utils.to_factored_line(pt3, pt4)
    #
    #     line1 = opencv_utils.to_factored_line(detected_lines[0][0], detected_lines[0][1])
    #     line2 = opencv_utils.to_factored_line(detected_lines[1][0], detected_lines[1][1])
    #
    #     upper_cross_point1 = opencv_utils.intersection(horizontal_upper_line, line1 )
    #     lower_cross_point1 = opencv_utils.intersection(horizontal_lower_line, line1)
    #     # print("Cross point1 upper{}".format(upper_cross_point1))
    #     # print("Cross point1 lower {}".format(lower_cross_point1))
    #
    #     upper_cross_point2 = opencv_utils.intersection(horizontal_upper_line, line2)
    #     lower_cross_point2 = opencv_utils.intersection(horizontal_lower_line, line2)
    #     # print("Cross point2 upper{}".format(upper_cross_point2))
    #     # print("Cross point2 lower {}".format(lower_cross_point2))
    #
    #     upper_mid_point = (np.array(upper_cross_point1) + np.array(upper_cross_point2)) / 2
    #     # print(upper_mid_point)
    #
    #     lower_mid_point = (np.array(lower_cross_point1) + np.array(lower_cross_point2)) / 2
    #     print(lower_mid_point)
    #
    #     detected_line = (upper_mid_point.astype(np.int), lower_mid_point.astype(np.int))
    #     detected_degree = (degrees[0] + degrees[1])/2
    #     detected_degree = opencv_utils.round_angle(DETECT_LINE_ANGLE_UNIT, detected_degree)
    #     diviation_distance = (diviation_distances[0] + diviation_distances[1])/2
    #     return [detected_line], [detected_degree], [diviation_distance]

    # return detected_lines, degrees, diviation_distances


def _filter_line_outliers(lines, max_degree, img_width, img_height):
    if len(lines) < 2:
        return lines

    lines_no_outliers = []
    for i, line in enumerate(lines):
        degree = lines[i].degree

        if abs(degree) < max_degree:
            lines_no_outliers.append(line)
    degree_median = np.median([item.degree for item in lines_no_outliers])
    degree_std_deviation = np.std([item.degree for item in lines_no_outliers])
    filtered_lines = []
    for i, line in enumerate(lines_no_outliers):
        degree = lines_no_outliers[i].degree
        if degree_median - 2 * degree_std_deviation < degree < degree_median + 2 * degree_std_deviation:
            filtered_lines.append(line)

    # Merge similar lane lines
    merged_lines = []

    compare_lines = []
    for item in filtered_lines:
        item.visited = False
        compare_lines.append(item)

    for item in compare_lines:
        # compare_lines[i][0]['visited'] = True
        item.visited = True
        merged_line = get_lane_line(item, compare_lines, img_width, img_height)
        if merged_line is None:
            continue
        merged_lines.append(merged_line)

    return merged_lines


def get_lane_line(line, compare_lines, img_width, img_height):
    LANE_MIN_DELTA_DEGREE = 0
    LANE_MAX_DELTA_DEGREE = 20
    LANE_MIN_DEVIATION_DISTANCE = 0
    LANE_MAX_DEVIATION_DISTANCE = 60

    for _line in compare_lines:
        if _line.visited:
            continue
        delta_degree = line.degree - _line.degree
        delta_distance = line.deviation_distance - _line.deviation_distance
        if LANE_MIN_DELTA_DEGREE < abs(delta_degree) < LANE_MAX_DELTA_DEGREE and LANE_MIN_DEVIATION_DISTANCE < abs(delta_distance) < LANE_MAX_DEVIATION_DISTANCE:
            # print("merge two lines into one")
            merged_degree = line.degree - delta_degree / 2
            merged_distance = line.deviation_distance - delta_distance / 2
            x1, y1, x2, y2 = merge_line(line, _line, img_width, img_height)
            if x1 is None:
                return None
            return LineXY(int(x1), int(y1), int(x2), int(y2), merged_degree, merged_distance)
    return None


def merge_line(_line1, _line2, img_width, img_height):
    pt1, pt2 = np.array([0, 0]), np.array([img_width, 0])
    pt3, pt4 = np.array([0, img_height]), np.array([img_width, img_height])
    horizontal_upper_line = opencv_utils.to_factored_line(pt1, pt2)
    horizontal_lower_line = opencv_utils.to_factored_line(pt3, pt4)

    line1 = opencv_utils.to_factored_line([_line1.x1, _line1.y1], [_line1.x2, _line1.y2])
    line2 = opencv_utils.to_factored_line([_line2.x1, _line2.y1], [_line2.x2, _line2.y2])

    upper_cross_point1 = opencv_utils.intersection(horizontal_upper_line, line1 )
    lower_cross_point1 = opencv_utils.intersection(horizontal_lower_line, line1)

    if upper_cross_point1 is None or lower_cross_point1 is None:
        return None, None, None, None
    # print("Cross point1 upper{}".format(upper_cross_point1))
    # print("Cross point1 lower {}".format(lower_cross_point1))

    upper_cross_point2 = opencv_utils.intersection(horizontal_upper_line, line2)
    lower_cross_point2 = opencv_utils.intersection(horizontal_lower_line, line2)
    # print("Cross point2 upper{}".format(upper_cross_point2))
    # print("Cross point2 lower {}".format(lower_cross_point2))

    upper_mid_point = (np.array(upper_cross_point1) + np.array(upper_cross_point2)) / 2
    # print(upper_mid_point)

    lower_mid_point = (np.array(lower_cross_point1) + np.array(lower_cross_point2)) / 2
    print(lower_mid_point)

    # detected_line = (upper_mid_point.astype(np.int), lower_mid_point.astype(np.int))
    return upper_mid_point[0], upper_mid_point[1], lower_mid_point[0], lower_mid_point[1]


def get_deviation_distance(img, x1, y1, x2, y2):
    """
    Get diviation distance of input line (x1, y1, x2, y2) with center vertical line of image
    Return -1 if no cross point
    :param img:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    p1 = np.array([int(img_width / 2), 0])
    p2 = np.array([int(img_width / 2), img_height])
    vertical_center_line = [p1, p2]
    horizontal_center_line = opencv_utils.to_factored_line(np.array([0, 420]), np.array([640, 420]))
    factored_line = opencv_utils.to_factored_line(np.array([x1, y1]), np.array([x2, y2]))
    cross_point = opencv_utils.intersection(horizontal_center_line, factored_line)
    d = -1
    if cross_point:
        d = opencv_utils.distance(vertical_center_line, cross_point)
    return d


def draw_detect_lines():
    img = cv2.imread('../images/7.jpg')

    roi = opencv_utils.get_roi_points(img, roi_heigh=130, roi_width_offset=25)
    roi_img = opencv_utils.get_roi_img(img, roi)
    # cv2.imshow('Crop ROI', roi_img)

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

    # cv2.imshow('Masked img', masked_img)

    # Blur image
    image_blured = cv2.bilateralFilter(masked_img, 13, 75, 75)
    # cv2.imshow('Bilateral filter Blured', image_blured)

    low_threshold = 50
    high_threshold = 150
    edges_img = cv2.Canny(image_blured, low_threshold, high_threshold)

    cv2.imshow('Edges', edges_img)

    detected_lines = get_detected_lines_and_degrees(edges_img)

    # Make a copy of the original image.
    img2 = np.copy(edges_img)

    if detected_lines:
        # Create a blank image that matches the original in size.
        line_image = np.zeros(
            (
                edges_img.shape[0],
                edges_img.shape[1],
                3
            ),
            dtype=np.uint8,
        )

        for line in detected_lines:
            print("Degree {}, distance deviation {}".format(line.degree, line.deviation_distance))
            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2)

            # cv2.line(line_image, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 255, 0), 1)
            cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 1)
            # Merge the image with the lines onto the original.
        img2 = cv2.addWeighted(roi_img, 0.8, line_image, 1.0, 0.0)

    cv2.imshow('Detected line deviation', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_detect_lines_img(img):
    """
    :param img: image in numpy array
    :return: image with drawn detected lines, detected_lines, degrees, deviation_distances from vertical center of screen
    """

    roi = opencv_utils.get_roi_points(img, roi_heigh=130, roi_width_offset=40)
    roi_img = opencv_utils.get_roi_img(img, roi)
    # cv2.imshow('Crop ROI', roi_img)

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

    # cv2.imshow('Masked img', masked_img)

    # Blur image
    image_blured = cv2.bilateralFilter(masked_img, 13, 75, 75)
    # cv2.imshow('Bilateral filter Blured', image_blured)

    low_threshold = 50
    high_threshold = 150
    edges_img = cv2.Canny(image_blured, low_threshold, high_threshold)

    # cv2.imshow('Edges', edges_img)

    detected_lines = get_detected_lines_and_degrees(edges_img)

    # Make a copy of the original image.
    img2 = np.copy(edges_img)

    if detected_lines:
        # Create a blank image that matches the original in size.
        line_image = np.zeros(
            (
                edges_img.shape[0],
                edges_img.shape[1],
                3
            ),
            dtype=np.uint8,
        )

        for line in detected_lines:
            # print("Degree {}".format(degree))
            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2)
            # cv2.line(line_image, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 255, 0), 1)
            cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 1)
            # Merge the image with the lines onto the original.
        img2 = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)

    return img2, detected_lines


class LineXY:
    def __init__(self, x1, y1, x2, y2, degree, deviation_distance):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.degree = degree
        self.deviation_distance = deviation_distance


if __name__ == '__main__':
    # realtime_lane_detect()
    draw_detect_lines()
