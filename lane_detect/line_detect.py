import cv2
import numpy as np
import opencv_utils

# img = cv2.imread('../images/1.jpg')
cap = cv2.VideoCapture(0)


def process_img():
    count = 0
    while True:
        _, img = cap.read()
        count = count + 1
        if count == 3:
            count = 0
        if count == 1 or count == 2:
            cv2.imshow('Detected line and deviation', img)
            continue
        img2, degree = get_detected_lines_and_degrees(img)
        if img2 is None:
            cv2.imshow('Detected line and deviation', img)
        else:
            cv2.putText(img2, "Deviation {} degree".format(degree), (150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

            cv2.imshow('Detected line and deviation', img2)

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
    :return:
    """
    DETECT_LINE_ANGLE_LIMIT=79
    DETECT_LINE_ANGLE_UNIT=5

    img_height = img.shape[0]
    img_width = img.shape[1]
    # This returns an array of r and theta values
    lines = cv2.HoughLines(img, rho=1, theta=np.pi / 20, threshold=30)
    if lines is None:
        return None, None, None
    print("Num of lines {}".format(lines[0].size))
    p1 = np.array([int(img_width/2), 0])
    p2 = np.array([int(img_width/2), img_height])
    vertical_center_line = [p1, p2]
    horizontal_center_line = opencv_utils.to_factored_line(np.array([0, 420]), np.array([640, 420]))
    degrees = []
    detected_lines = []
    diviation_distances = []

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
            print("Degree {}".format(degree))

            if np.abs(degree) > DETECT_LINE_ANGLE_LIMIT:
                continue
            degree = opencv_utils.round_angle(DETECT_LINE_ANGLE_UNIT, degree)

            # factored_line = opencv_utils.to_factored_line(np.array([x1, y1]), np.array([x2, y2]))
            # cross_point = opencv_utils.intersection(horizontal_center_line, factored_line)
            d = get_deviation_distance(img, x1, y1, x2, y2)
            if d != -1:
                diviation_distances.append(d)
                detected_lines.append(((x1, y1), (x2, y2)))
                degrees.append(degree)
                print("Line: (x1, y1) {}, (x2, y2) {}, r {}, degree {}, deviation distance {}".format((x1, y1), (x2, y2), r, degree, d))

    # Process when there is exactly two lines detected
    if len(detected_lines) == 2:
        print("merge two lines into one")
        pt1, pt2 = np.array([0, 0]), np.array([img_width, 0])
        pt3, pt4 = np.array([0, img_height]), np.array([img_width, img_height])
        horizontal_upper_line = opencv_utils.to_factored_line(pt1, pt2)
        horizontal_lower_line = opencv_utils.to_factored_line(pt3, pt4)

        line1 = opencv_utils.to_factored_line(detected_lines[0][0], detected_lines[0][1])
        line2 = opencv_utils.to_factored_line(detected_lines[1][0], detected_lines[1][1])

        upper_cross_point1 = opencv_utils.intersection(horizontal_upper_line, line1 )
        lower_cross_point1 = opencv_utils.intersection(horizontal_lower_line, line1)
        print("Cross point1 upper{}".format(upper_cross_point1))
        print("Cross point1 lower {}".format(lower_cross_point1))

        upper_cross_point2 = opencv_utils.intersection(horizontal_upper_line, line2)
        lower_cross_point2 = opencv_utils.intersection(horizontal_lower_line, line2)
        print("Cross point2 upper{}".format(upper_cross_point2))
        print("Cross point2 lower {}".format(lower_cross_point2))

        upper_mid_point = (np.array(upper_cross_point1) + np.array(upper_cross_point2)) / 2
        print(upper_mid_point)

        lower_mid_point = (np.array(lower_cross_point1) + np.array(lower_cross_point2)) / 2
        print(lower_mid_point)

        detected_line = (upper_mid_point.astype(np.int), lower_mid_point.astype(np.int))
        detected_degree = (degrees[0] + degrees[1])/2
        detected_degree = opencv_utils.round_angle(DETECT_LINE_ANGLE_UNIT, detected_degree)
        diviation_distance = (diviation_distances[0] + diviation_distances[1])/2
        return [detected_line], [detected_degree], [diviation_distance]

    return detected_lines, degrees, diviation_distances


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
    img = cv2.imread('../images/8.jpg')

    roi = opencv_utils.get_roi_points(img, roi_heigh=130, roi_width_offset=5)
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

    detected_lines, degrees, deviation_distances = get_detected_lines_and_degrees(edges_img)

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

        for degree, line in zip(degrees, detected_lines):
            print("Degree {}".format(degree))
            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2)
            cv2.line(line_image, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 255, 0), 1)
            # Merge the image with the lines onto the original.
        img2 = cv2.addWeighted(roi_img, 0.8, line_image, 1.0, 0.0)

    cv2.imshow('Detected line deviation', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # process_img()
    # draw_detect_line()
    draw_detect_lines()
    # check_color_from_file()