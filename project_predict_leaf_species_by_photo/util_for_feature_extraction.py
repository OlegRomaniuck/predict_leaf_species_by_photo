import os
import cv2
import numpy as np
import pandas as pd
import mahotas as mt

# health = 1 unhelath = 0
dict_leaf = {"Tomato": 1, "Grape": 2, "Apple": 3, "Pepper": 4, 'Strawberry': 5, 'Squash': 6, 'Corn': 7, 'Peach': 8,
             'Soybean': 9, 'Potato': 10, 'Raspberry': 11, 'Cherry': 12, 'Orange': 13, 'Blueberry': 14}


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


dataset = extract_feature_color("data_set/train")
dataset.to_csv("leaf_features_color_train.csv")
dataset = extract_feature_color("data_set/val")
dataset.to_csv("leaf_features_color_valid.csv")
dataset = extract_feature_color( "data_set/test")
dataset.to_csv("leaf_features_color_test.csv")
