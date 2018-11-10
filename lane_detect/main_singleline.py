import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2


def process_img():
    # reading in an image
    image = mpimg.imread('img/lan1.jpg')
    # image = mpimg.imread('img/img_a2.jpg')

    height = image.shape[0]
    width = image.shape[1]

    # Convert to grayscale here.
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 200, 300)

    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    # cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32),)

    # Detect line
    lines = cv2.HoughLinesP(
        cannyed_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    # print(lines)
    line_image = draw_lines(image, lines)

    cv2.imshow('Line img', line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    # channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * 3

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_image = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)
    # Return the modified image.
    return img


if __name__ == '__main__':
    process_img()