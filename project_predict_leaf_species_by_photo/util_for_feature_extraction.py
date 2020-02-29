import os
import cv2
import numpy as np
import pandas as pd
import mahotas as mt
from skimage import feature
from scipy import ndimage
from matplotlib import pyplot as plt


# health = 1 unhelath = 0
dict_leaf = {"Tomato": 1, "Grape": 2, "Apple": 3, "Pepper": 4, 'Strawberry': 5, 'Squash': 6, 'Corn': 7, 'Peach': 8,
             'Soybean': 9, 'Potato': 10, 'Raspberry': 11, 'Cherry': 12, 'Orange': 13, 'Blueberry': 14}

x1 = y1 = x2 = y2 = flag = angle = 0


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")

        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        # hist /= (hist.sum() + eps)
        return hist


def resizeImage(image, size):
    # cv2.imshow('Resized', cv2.resize(image, (size,size), interpolation=cv2.INTER_CUBIC))
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def rotateImage(image, angle):
    # cv2.imshow('Rotated', ndimage.rotate(image, angle, cval=256))
    return ndimage.rotate(image, angle, cval=256)


def rotationAngle(image):
    # cv2.imshow('Image', image)
    # cv2.setMouseCallback('Image', mouseClickEvent)
    while (1):
        if cv2.waitKey(0): cv2.destroyAllWindows();break
    return angle


def obtainHu(image):
    image = cv2.bitwise_not(image)
    # cv2.imshow("Bitwise Not", image)
    huInvars = cv2.HuMoments(cv2.moments(image)).flatten()  # Obtain hu moments from normalised moments in an array
    huInvars = -np.sign(huInvars) * np.log10(np.abs(huInvars))
    # huInvars /= huInvars.sum()
    return huInvars


def obtainHuMoments(image_path, image):  # Obtains hu moments from image at path location
    hu = obtainHu(image)
    image = cv2.imread(image_path, 0)
    lbp = LocalBinaryPatterns(24, 8)
    hist = lbp.describe(image)

    return hu, hist


def getAFR(length, width, area, perimeter):
    aspect_ratio = length / width
    form_factor = 4 * 3.14159265358 * area / perimeter
    rectangularity = length * width / area
    return aspect_ratio, form_factor, rectangularity


def findLeafContour(contours):
    maxArea = 0
    contour_index = 0;
    for i in range(1, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxArea:
            maxArea = area
            contour_index = i
    return contour_index


def leafLW(image):
    l, w = image.shape[:2]
    return l - 20, w - 20


def leafAP(image):
    value = 230
    retval, img = cv2.threshold(image, value, 255, type=cv2.THRESH_BINARY)
    contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    while (len(contours) <= 1):
        value = value - 5
        retval, img = cv2.threshold(image, value, 255, type=cv2.THRESH_BINARY)
        contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    leaf_contour_index = findLeafContour(contours)
    leaf_contour = contours[leaf_contour_index]
    cv2.drawContours(img, contours, leaf_contour_index, (0, 0, 255), 4)
    # cv2.imshow('contours', img)
    area = cv2.contourArea(leaf_contour)
    perimeter = cv2.arcLength(leaf_contour, True)
    return area, perimeter


def croppedImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    # cv2.imshow("Grey", gray)
    blur = cv2.medianBlur(gray, 5)  # make the image blur
    # cv2.imshow("Blur", blur)
    retval, thresh_gray = cv2.threshold(blur, 200, 255, type=cv2.THRESH_BINARY)  # threshold to get just the leaf
    # cv2.imshow("Thresh", thresh_gray)
    # find where the leaf is and make a cropped region
    points = np.argwhere(thresh_gray == 0)  # find where the black pixels are
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    min_x = min(x for (x, y) in points)
    min_y = min(y for (x, y) in points)
    max_x = max(x for (x, y) in points)
    max_y = max(y for (x, y) in points)
    crop = blur[min_y - 10:max_y + 10, min_x - 10:max_x + 10]  # create a cropped region of the blur image
    retval, thresh_crop = cv2.threshold(crop, 200, 255, type=cv2.THRESH_BINARY)
    # cv2.imshow('Thresh and Cropped', thresh_crop)
    return thresh_crop


def getLWAP(path):
    img = cv2.imread(path)
    while img.shape[1] >= 800 or img.shape[2] >= 800:
        img = cv2.resize(img, None, fx=0.9, fy=0.9)
    angle = rotationAngle(img)
    img = rotateImage(img, angle)
    img = croppedImage(img)
    length, width = leafLW(img)
    # print(length, width)
    # img = resizeImage(img, 800)
    area, perimeter = leafAP(img)
    # print(area, perimeter)
    return img, length, width, area, perimeter


def extract_feature_color(path):
    names = ['area', 'perimeter', 'physiological_length', 'physiological_width', 'aspect_ratio', 'rectangularity',
             'circularity', 'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b', 'contrast', 'correlation',
             'inverse_difference_moments', 'entropy', 'laplacian_var', 'sobelx_var', 'sobely_var', 'sobelx8u_var',
             'sobelx64f_var', 'sobel_8u_var', 'type_leaf', 'name', 'health']
    df = pd.DataFrame([], columns=names)
    for subdir, dirs, files in os.walk(path):
        for file in files:
            path_to_file = os.path.join(subdir, file)
            print(path_to_file)
            name = str(file)
            type_leaf = str(file).split('_')[0]
            type_leaf = dict_leaf.get(type_leaf)
            health = 1
            if "unhealthy" in file:
                health = 0
            main_img = cv2.imread(path_to_file)
            laplacian_var = cv2.Laplacian(main_img, cv2.CV_64F).var()
            sobelx_var = cv2.Sobel(main_img, cv2.CV_64F, 1, 0, ksize=5).var()
            sobely_var = cv2.Sobel(main_img, cv2.CV_64F, 0, 1, ksize=5).var()
            sobelx8u_var = cv2.Sobel(main_img, cv2.CV_8U, 1, 0, ksize=5).var()
            sobelx64f = cv2.Sobel(main_img, cv2.CV_64F, 1, 0, ksize=5)
            sobelx64f_var = sobelx64f.var()
            abs_sobel64f = np.absolute(sobelx64f)
            sobel_8u_var = np.uint8(abs_sobel64f).var()

            # Preprocessing
            img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
            gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gs, (25, 25), 0)
            ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((50, 50), np.uint8)
            closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
            # Shape features
            contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if area == 0:
                area = 1
            rectangularity = w * h / area
            circularity = ((perimeter) ** 2) / area

            # Color features
            red_channel = img[:, :, 0]
            green_channel = img[:, :, 1]
            blue_channel = img[:, :, 2]
            blue_channel[blue_channel == 255] = 0
            green_channel[green_channel == 255] = 0
            red_channel[red_channel == 255] = 0

            red_mean = np.mean(red_channel)
            green_mean = np.mean(green_channel)
            blue_mean = np.mean(blue_channel)

            red_std = np.std(red_channel)
            green_std = np.std(green_channel)
            blue_std = np.std(blue_channel)

            # Texture features
            textures = mt.features.haralick(gs)
            ht_mean = textures.mean(axis=0)
            contrast = ht_mean[1]
            correlation = ht_mean[2]
            inverse_diff_moments = ht_mean[4]
            entropy = ht_mean[8]

            vector = [area, perimeter, w, h, aspect_ratio, rectangularity, circularity, red_mean, green_mean,
                      blue_mean, red_std, green_std, blue_std, contrast, correlation, inverse_diff_moments, entropy,
                      laplacian_var, sobelx_var, sobely_var, sobelx8u_var,
                      sobelx64f_var, sobel_8u_var,
                      type_leaf, name, health]

            df_temp = pd.DataFrame([vector], columns=names)
            df = df.append(df_temp)
    return df


# dataset = extract_feature_color(
#     "/data_set/train")
# print(dataset.shape)
# dataset.to_csv("leaf_features_color_train.csv")

# dataset = extract_feature_color(
#     "/data_set/val")
# print(dataset.shape)
# dataset.to_csv("leaf_features_color_valid.csv")
# dataset = extract_feature_color(
#     "/data_set/test")
# print(dataset.shape)
# dataset.to_csv("leaf_features_color_test.csv")


