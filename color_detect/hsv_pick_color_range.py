import cv2
import numpy as np

image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default
hsv_lower = None, None, None
hsv_upper = None, None, None

# mouse callback function
def pick_color(event,x,y,flags,param):
    global hsv_lower
    global hsv_upper
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(pixel, lower, upper)

        if not hsv_lower[0] and not hsv_upper[0]:
            hsv_lower = lower
            hsv_upper = upper
        else:
            if hsv_lower[0] > lower[0]:
                hsv_lower[0] = lower[0]
            if hsv_lower[1] > lower[1]:
                hsv_lower[1] = lower[1]
            if hsv_lower[2] > lower[2]:
                hsv_lower[2] = lower[2]

            if hsv_upper[0] < upper[0]:
                hsv_upper[0] = upper[0]
            if hsv_upper[1] < upper[1]:
                hsv_upper[1] = upper[1]
            if hsv_upper[2] < upper[2]:
                hsv_upper[2] = upper[2]

            print("Your color range change: ")
            print(hsv_lower)
            print(hsv_upper)
            image_mask = cv2.inRange(image_hsv,hsv_lower, hsv_upper)
            cv2.imshow("mask",image_mask)


def main():
    global image_hsv, pixel # so we can use it in mouse callback

    image_src = cv2.imread("images/2.jpg")
    cv2.imshow("bgr",image_src)

    ## NEW ##
    cv2.namedWindow('hsv')
    cv2.setMouseCallback('hsv', pick_color)

    # now click into the hsv img , and look at values:
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv",image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()