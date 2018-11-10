import numpy as np
import cv2


def main_demo():
    img = cv2.imread('../images/1.jpg')
    roi = get_roi(img)
    print("ROI: {}".format(roi))
    # cv2.imshow('origin img', img)

    roi_img = get_roi_img(img, roi)
    cv2.imshow('ROI img', roi_img)

    #Blur image
    image_blured = cv2.bilateralFilter(roi_img, 13, 75, 75)
    # cv2.imshow('Bilateral filter Blured', image_blured)

    image_unsharp = cv2.addWeighted(image_blured, 1.5, image_blured, -0.5, 0, roi_img)
    # cv2.imshow('Bilateral filter unsharp addWeighted Blured', image_unsharp)

    image_clached = clache(image_unsharp)
    cv2.imshow('Clache Blured', image_clached)

    # ----------------------------- HOUGH ---------------------------

    left_lns, right_lns, median_left_f, median_right_f = detect(
        image_clached, lane_color_value=5, prev_left_ms = [], prev_left_bs = [], prev_right_ms = [], prev_right_bs = [])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def weighted_img(img, initial_img, alpha=0.8, beta=1., lbda=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, lbda)


def detect(image, lane_color_value, prev_left_ms, prev_left_bs, prev_right_ms, prev_right_bs):
    """???I want black lane"""
    houghed_lns = hough_lines(image, lane_color_value)

    if houghed_lns is None:
        return [None, None, None, None]

    filtered_left_lns, filtered_right_lns = left_right_lines(houghed_lns)
    median_left_f = median(filtered_left_lns, prev_left_ms, prev_left_bs)
    median_right_f = median(filtered_right_lns, prev_right_ms, prev_right_bs)

    if median_left_f is None or median_right_f is None:
        return detect(image, lane_color_value - 5, prev_left_ms, prev_left_bs, prev_right_ms, prev_right_bs)
    else:
        return [filtered_left_lns, filtered_right_lns, median_left_f, median_right_f]


def hough_lines(img, lane_color_value):
    """???I want black lane"""
    HSV_BLACK_UPPER_BOUNDARY = 150
    if lane_color_value > HSV_BLACK_UPPER_BOUNDARY:
        return None

    image_masked = color_threshold(img, lane_color_value)
    image_edges = edges(image_masked)
    houghed_lns = cv2.HoughLinesP(image_edges, 2, np.pi / 180, 50, np.array([]), 20, 100)

    if houghed_lns is None:
        return hough_lines(img, lane_color_value - 5)

    return houghed_lns


def filter_by_color_ranges(img, color_ranges):
    result = np.zeros_like(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for color_range in color_ranges:
        color_bottom = color_range[0]
        color_top = color_range[1]

        if color_bottom[0] == color_bottom[1] == color_bottom[2] and color_top[0] == color_top[1] == color_top[2]:
            mask = binary_gray_mask(gray_img, color_range)
        else:
            mask = binary_hsv_mask(hsv_img, color_range)

        masked_img = binary_mask_apply(img, mask)
        result = cv2.addWeighted(masked_img, 1.0, result, 1.0, 0.0)

    return result

def clache(img):
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    y_clache = cl.apply(y)
    img_yuv = cv2.merge((y_clache, u, v))

    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def get_roi(img):
    """Return list of points for region of interest"""
    height = img.shape[0]

    roi_heigh = 130
    roi_width_offset = 50
    left_top_roi, right_top_roi, right_bottom_roi, left_bottom_roi = (roi_width_offset, height - roi_heigh), (640 - roi_width_offset, height - roi_heigh), (640 - roi_width_offset, height), (roi_width_offset, height)
    l1 = [left_top_roi, right_top_roi, right_bottom_roi, left_bottom_roi]
    roi = [np.array(l1, dtype=np.int32)]
    return roi


def get_roi_img(img, vertices):
    """Return region of interest image from a numpy array of points"""
    mask = np.zeros_like(img)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

if __name__ == '__main__':
    main_demo()