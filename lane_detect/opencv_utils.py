import numpy as np
import cv2
from matplotlib.path import Path


def distance(line, p3):
    d = np.cross(line[1] - line[0], p3 - line[0]) / np.linalg.norm(line[1] - line[0])
    return np.abs(d)


def get_roi_points(img, roi_heigh=130, roi_width_offset=50):
    """Return list of points in numpy array for region of interest"""
    height = img.shape[0]
    width = img.shape[1]
    left_top_roi, right_top_roi, right_bottom_roi, left_bottom_roi = (roi_width_offset, height - roi_heigh), (width - roi_width_offset, height - roi_heigh), (width - roi_width_offset, height), (roi_width_offset, height)
    l1 = [left_top_roi, right_top_roi, right_bottom_roi, left_bottom_roi]
    roi = [np.array(l1, dtype=np.int32)]
    return roi


def get_roi_img(img, roi_points):
    """
    @img: image in numpy array
    @roi_points: numpy array of points
    Crop region of interest image
    """
    mask = np.zeros_like(img)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, roi_points, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def to_factored_line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False


def round_angle(round_degree_unit, degree):
    if degree == 0:
        return 0
    d2 = abs(degree) // round_degree_unit
    d4 = round_degree_unit * d2 * (degree / abs(degree))
    return int(d4)


def round_angle_tests():
    round_degree_unit = 10
    degree = 9
    print(round_angle(round_degree_unit, 9))
    print(round_angle(round_degree_unit, 0))
    print(round_angle(round_degree_unit, 10))
    print(round_angle(round_degree_unit, 13))
    print(round_angle(round_degree_unit, 19))
    print(round_angle(round_degree_unit, 20))
    print(round_angle(round_degree_unit, 21))

    print()

    print(round_angle(round_degree_unit, -9))
    print(round_angle(round_degree_unit, -10))
    print(round_angle(round_degree_unit, -13))
    print(round_angle(round_degree_unit, -19))
    print(round_angle(round_degree_unit, -20))
    print(round_angle(round_degree_unit, -21))

if __name__ == '__main__':
    # p1 = np.array([0, 0])
    # p2 = np.array([10, 10])
    # p3 = np.array([5, 7])
    # line = [p1, p2]
    # print("distance: {}".format(distance(line, p3)))

    # p1, p2 = np.array([0, 420]), np.array([640, 420])
    # p3, p4 = np.array([0, 350]), np.array([308, 479])
    #
    # line1 = to_factored_line(p1, p2)
    # line2 = to_factored_line(p3, p4)
    #
    # cross_point = intersection( line1, line2)
    # print("Cross point {}".format(cross_point))

    round_angle_tests()