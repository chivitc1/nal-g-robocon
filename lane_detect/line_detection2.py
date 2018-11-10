import numpy as np
import cv2

from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main_demo():
    # img = cv2.imread('../images/1.jpg')
    # roi = get_roi(img)

    image = mpimg.imread('test_images/challenge1.jpg')

    # image = mpimg.imread('../images/1.jpg')
    image, _, _ = image_pipeline(image, should_plot=True)
    show_image(image)
    plt.show()


def weighted_img(img, initial_img, alpha=0.8, beta=1., lbda=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, lbda)


def stack_images(img_a, img_b):
    return np.hstack((img_a, img_b))

################Plot use
def show_image(img):
    plt.figure(figsize = (15, 10))
    plt.imshow(img)


def show_gray_image(img):
    plt.figure(figsize = (15, 10))
    plt.imshow(img, cmap='gray')


def plot_start():
    plt.figure(figsize = (15, 10))


def plot_lines(lines, color):
    for line in lines:
        for x1, y1, x2, y2 in line:
            plt.plot((x1, x2), (y1, y2), color)


def plot_function(f, xs):
    for x in range(xs[0], xs[len(xs) - 1]):
        plt.plot(x, f(x), 'g2')
################End Plot use


def get_roi(img):
    """Return list of points for region of interest"""
    height = img.shape[0]

    roi_heigh = 130
    roi_width = 10
    left_top_roi, right_top_roi, right_bottom_roi, left_bottom_roi = (roi_width, height - roi_heigh), (640 - roi_width, height - roi_heigh), (640 - roi_width, height), (roi_width, height)
    roi = [left_top_roi, right_top_roi, right_bottom_roi, left_bottom_roi]
    return roi

#######


def get_roi_img(img, vertices):
    """Return region of interest image from a numpy array of points"""
    print("vertices type {}, value: {}".format(type(vertices), vertices))
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


def crop_roi(img, top_left, top_right, bottom_right, bottom_left):
    """Return region of interest image from 4 points"""
    roi = [np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)]
    print("NP array ROI: {}".format(roi))
    return get_roi_img(img, roi)


def crop_by_ref(img, ref_width, ref_height, ref_top_x, ref_top_y, ref_bot_x, ref_bot_y):
    width = img.shape[1]
    image_height = img.shape[0]
    middle_x = int(width / 2)
    image_offset_bottom_x = int(width * ref_bot_x / ref_width)
    image_offset_bottom_y = int(image_height * ref_bot_y / ref_height)
    image_offset_top_x = int(width * ref_top_x / ref_width)
    image_offset_top_y = int(image_height * ref_top_y / ref_height)
    top_left = [middle_x - image_offset_top_x, image_offset_top_y]
    top_right = [middle_x + image_offset_top_x, image_offset_top_y]
    bottom_right = [width - image_offset_bottom_x, image_offset_bottom_y]
    bottom_left = [image_offset_bottom_x, image_offset_bottom_y]

    return crop_roi(img, top_left, top_right, bottom_right, bottom_left)


def crop(img, bottom_offset = 0):
    """????"""
    ref_width = 960
    ref_height = 540
    ref_top_x = 50
    ref_top_y = 300
    ref_bottom_x = 70
    ref_bottom_y = 540 - bottom_offset

    return crop_by_ref(img, ref_width, ref_height, ref_top_x, ref_top_y, ref_bottom_x, ref_bottom_y)


def equalize_histogram(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def clache(img):
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    y_clache = cl.apply(y)
    img_yuv = cv2.merge((y_clache, u, v))

    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def biliteral(img):
    return cv2.bilateralFilter(img, 13, 75, 75)


def unsharp_mask(image, blured):
    return cv2.addWeighted(blured, 1.5, blured, -0.5, 0, image)


def edges(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = np.median(gray_img)
    sigma = 0.33
    lower = int(max(150, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return canny(gray_img, lower, upper)


def binary_hsv_mask(img, color_range):
    lower = np.array(color_range[0])
    upper = np.array(color_range[1])

    return cv2.inRange(img, lower, upper)


def binary_gray_mask(img, color_range):
    return cv2.inRange(img, color_range[0][0], color_range[1][0])


def binary_mask_apply(img, binary_mask):
    masked_image = np.zeros_like(img)

    for i in range(3):
        masked_image[:, :, i] = binary_mask.copy()

    return masked_image


def binary_mask_apply_color(img, binary_mask):
    return cv2.bitwise_and(img, img, mask=binary_mask)


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


def color_threshold(img, white_value):
    """??? I want black lane line"""
    white = [[white_value, white_value, white_value], [255, 255, 255]]
    yellow = [[80, 90, 90], [120, 255, 255]]

    return filter_by_color_ranges(img, [white, yellow])


def draw_lanes(height, width, left_x, right_x, intersect_pt):
    line_img = np.zeros((height, width, 3), dtype=np.uint8)

    if left_x != 0 and intersect_pt != None:
        cv2.line(line_img, (left_x, height), (intersect_pt[0], intersect_pt[1]), [255, 0, 0], 5)

    if right_x != 0 and intersect_pt != None:
        cv2.line(line_img, (intersect_pt[0], intersect_pt[1]), (right_x, height), [0, 255, 0], 5)

    return line_img


def hough_lines(img, white_value):
    """???I want black lane"""
    if white_value < 150:
        return None

    image_masked = color_threshold(img, white_value)
    image_edges = edges(image_masked)
    houghed_lns = cv2.HoughLinesP(image_edges, 2, np.pi / 180, 50, np.array([]), 20, 100)

    if houghed_lns is None:
        return hough_lines(img, white_value - 5)

    return houghed_lns


def filter_lines_outliers(lines, slopes, min_slope=0.5, max_slope=0.9):
    if len(lines) < 2:
        return lines

    lines_no_outliers = []
    slopes_no_outliers = []

    for i, line in enumerate(lines):
        slope = slopes[i]

        if min_slope < abs(slope) < max_slope:
            lines_no_outliers.append(line)
            slopes_no_outliers.append(slope)

    slope_median = np.median(slopes_no_outliers)
    slope_std_deviation = np.std(slopes_no_outliers)
    filtered_lines = []

    for i, line in enumerate(lines_no_outliers):
        slope = slopes_no_outliers[i]

        if slope_median - 2 * slope_std_deviation < slope < slope_median + 2 * slope_std_deviation:
            filtered_lines.append(line)

    return filtered_lines


def left_right_lines(lines):
    lines_all_left = []
    lines_all_right = []
    slopes_left = []
    slopes_right = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)

            if slope > 0:
                lines_all_right.append(line)
                slopes_right.append(slope)
            else:
                lines_all_left.append(line)
                slopes_left.append(slope)

    filtered_left_lns = filter_lines_outliers(lines_all_left, slopes_left)
    filtered_right_lns = filter_lines_outliers(lines_all_right, slopes_right)

    return filtered_left_lns, filtered_right_lns


def median(lines, prev_ms, prev_bs):
    if prev_ms is None:
        prev_ms = []
        prev_bs = []

    xs = []
    ys = []
    xs_med = []
    ys_med = []
    m = 0
    b = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs += [x1, x2]
            ys += [y1, y2]

    if len(xs) > 2 and len(ys) > 2:
        #         m, b = np.polyfit(xs, ys, 1)
        m, b, r_value_left, p_value_left, std_err = linregress(xs, ys)

        if len(prev_ms) > 0:
            prev_ms.append(m)
            prev_bs.append(b)
        else:
            return np.poly1d([m, b])

    if len(prev_ms) > 0:
        return np.poly1d([np.average(prev_ms), np.average(prev_bs)])
    else:
        return None


def intersect(f_a, f_b):
    if f_a is None or f_b is None:
        return None

    equation = f_a.coeffs - f_b.coeffs
    x = -equation[1] / equation[0]
    y = np.poly1d(f_a.coeffs)(x)
    x, y = map(int, [x, y])

    return [x, y]


def detect(image, white_value, prev_left_ms, prev_left_bs, prev_right_ms, prev_right_bs):
    """???I want black lane"""
    houghed_lns = hough_lines(image, white_value)

    if houghed_lns is None:
        return [None, None, None, None]

    filtered_left_lns, filtered_right_lns = left_right_lines(houghed_lns)
    median_left_f = median(filtered_left_lns, prev_left_ms, prev_left_bs)
    median_right_f = median(filtered_right_lns, prev_right_ms, prev_right_bs)

    if median_left_f is None or median_right_f is None:
        return detect(image, white_value - 5, prev_left_ms, prev_left_bs, prev_right_ms, prev_right_bs)
    else:
        return [filtered_left_lns, filtered_right_lns, median_left_f, median_right_f]


def image_pipeline(image, prev_left_ms=[], prev_left_bs=[], prev_right_ms=[],
                   prev_right_bs=[], bottom_offset=0, should_plot=False):
    height = image.shape[0]
    width = image.shape[1]

    image_cropped = crop(image, bottom_offset)
    image_blured = biliteral(image_cropped)
    image_unsharp = unsharp_mask(image_cropped, image_blured)
    image_clached = clache(image_unsharp)

    # ----------------------------- HOUGH ---------------------------
    left_lns, right_lns, median_left_f, median_right_f = detect(
        image_clached, 220, prev_left_ms, prev_left_bs, prev_right_ms, prev_right_bs)

    if median_left_f is None and median_right_f is None:
        return [image, None, None]

    intersect_pt = intersect(median_left_f, median_right_f)

    if intersect_pt is None:
        return [image, median_left_f, median_right_f]

    left_x = int((median_left_f - height).roots)
    right_x = int((median_right_f - height).roots)

    # ----------------------------- DRAW ---------------------------

    if should_plot:
        plot_start()
        plot_lines(left_lns, 'r')
        plot_lines(right_lns, 'b')

        if median_left_f is not None:
            plot_function(median_left_f, [left_x, intersect_pt[0]])

        if median_right_f is not None:
            plot_function(median_right_f, [intersect_pt[0], right_x])

    lanes = draw_lanes(height, width, left_x, right_x, intersect_pt)
    return [weighted_img(lanes, image), median_left_f, median_right_f]



if __name__ == '__main__':
    main_demo()