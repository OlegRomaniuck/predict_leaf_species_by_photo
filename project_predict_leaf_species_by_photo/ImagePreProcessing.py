import os

import cv2
import numpy as np
from scipy import ndimage
import math
from matplotlib import pyplot as plt
from DIPLOMA.my_dipl.kursovaya.utils import *
from DIPLOMA.my_dipl.kursovaya.background_marker import *


x1=y1=x2=y2=flag=angle=0

def mouseClickEvent(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        global flag,angle
        global x1,x2,y1,y2
        flag= flag+1
        if flag == 1:
            x1=x
            y1=y
            # print(x,y)
        elif flag == 2:
            x2=x
            y2=y
            # print(x,y)
            slope = ((y2-y1)/(x2-x1))
            angle = math.atan(slope)*180*7/22

def croppedImage(image):
    blur = create_inverted_image(image)
    # cv2.imshow("ivnered_smother", blur)
    # cv2.waitKey(0)
    retval, thresh_gray = cv2.threshold(blur, 200, 255, type=cv2.THRESH_BINARY)  # threshold to get just the leaf

    # find where the leaf is and make a cropped region
    points = np.argwhere(thresh_gray == 0)  # find where the black pixels are
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    min_x = min(x for (x, y) in points)
    min_y = min(y for (x, y) in points)
    max_x = max(x for (x, y) in points)
    max_y = max(y for (x, y) in points)
    # crop = blur[min_y-10:max_y+10, min_x-10:max_x+10]
    crop = blur[min_y:max_y, min_x:max_x]  # create a cropped region of the blur image
    retval, thresh_crop = cv2.threshold(crop, 200, 255, type=cv2.THRESH_BINARY)
    # cv2.imshow('Thresh and Cropped', thresh_crop)
    # cv2.waitKey(0)
    return thresh_crop

def resizeImage(image, size):
    # cv2.imshow('Resized', cv2.resize(image, (size,size), interpolation=cv2.INTER_CUBIC))
    return cv2.resize(image, (size,size), interpolation=cv2.INTER_CUBIC)

def rotateImage(image, angle):
    # cv2.imshow('Rotated', ndimage.rotate(image, angle, cval=256))
    return ndimage.rotate(image, angle, cval=256)

def rotationAngle(image):
    # cv2.imshow('Image', image)
    # cv2.setMouseCallback('Image', mouseClickEvent)
    while (1):
        if cv2.waitKey(0): cv2.destroyAllWindows();break
    return angle

def generate_background_marker(image):
    original_image = image

    marker = np.full((original_image.shape[0], original_image.shape[1]), True)

        # update marker based on vegetation color index technique
    color_index_marker(index_diff(original_image), marker)

        # update marker to remove blues
        # remove_blues(original_image, marker)

    return original_image, marker

def create_inverted_image(image):
    filling_mode = FILL['THRESHOLD']
    original, marker = generate_background_marker(image)
    # set up binary image for futher processing
    bin_image = np.zeros((original.shape[0], original.shape[1]))
    bin_image[marker] = 255
    bin_image = bin_image.astype(np.uint8)
    largest_mask = select_largest_obj(bin_image, fill_mode=filling_mode, smooth_boundary=True)
    image = largest_mask
    # cv2.imshow("before inverted", image)
    # cv2.waitKey(0)
    image = cv2.bitwise_not(image)
    return image

