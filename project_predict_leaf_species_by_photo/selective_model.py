import os
import pickle
import subprocess
import pandas as pd
import mahotas as mt
from PIL import Image
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from ImageFeature import *

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
dict_leaf = {"Tomato": 1, "Grape": 2, "Apple": 3, "Pepper": 4, 'Strawberry': 5, 'Squash': 6, 'Corn': 7, 'Peach': 8,
             'Soybean': 9, 'Potato': 10, 'Raspberry': 11, 'Cherry': 12, 'Orange': 13, 'Blueberry': 14}


def feature_extractor_simple(image):
    names = ['area', 'perimeter', 'physiological_length', 'physiological_width', 'aspect_ratio', 'rectangularity',
             'circularity', 'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b', 'contrast', 'correlation',
             'inverse_difference_moments', 'entropy', 'laplacian_var', 'sobelx_var', 'sobely_var', 'sobelx8u_var',
             'sobelx64f_var', 'sobel_8u_var', 'health']

    health = 1
    main_img = cv2.imread(image)

    # gradienst
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
    # close holes
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
              sobelx64f_var, sobel_8u_var, health]

    df = pd.DataFrame([vector], columns=names)

    return df


def convert_into_specie(numbr, add=None):
    numb = numbr if add is None else numbr + 1
    for key, value in dict_leaf.items():
        if value == numb:
            return key


def get_model(model_name):
    if model_name == "random_forest":
        return random_forest_predictor
    elif model_name == "xgboost":
        return xgboost_predictior


def global_predictor(image, model_name):
    species = "Can not recognize leaf species, statistically you should choose apple or orange = ) "
    try:
        if model_name in ["random_forest", "xgboost"]:
            X_val = feature_extractor_simple(image)
            predictions = get_model(model_name)(X_val)
            species = convert_into_specie(predictions[0], add=1)
            return species
        else:
            image_name = vgg_feature_extractor(image)
            species = vgg_predictor(image_name)

        return species
    except:
        return species


def vgg_feature_extractor(img_path):
    segmented_image_name = segment_image(img_path)

    return segmented_image_name


def vgg_predictor(img_path):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VGG_all_100p_94.h5")
    img_target_size = (100, 100)

    model = load_model(model_path)

    # get image as array and resize it if necessary
    pil_img = Image.open(img_path)
    if pil_img.size != img_target_size:
        pil_img = pil_img.resize(img_target_size)

    img = image.img_to_array(pil_img)

    # if alpha channel found, discard it
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # preprocess image
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img).flatten()

    # get predictions index sorted based on the best predictions
    value_ = preds.argsort()
    sorted_preds_index = value_[::-1]

    # all species and supported species names
    SPECIES = ["Apple", "Soybean", 'Blueberry', 'Cherry', 'Corn', "Grape", 'grapefruit', 'Orange', 'Peach',
               "Pepper", 'Potato', 'Raspberry', 'Sorghum', 'Soybean', 'Squash', 'Strawberry', 'sugarcane', "Tomato"]
    return SPECIES[sorted_preds_index[0]]


def segment_image(img_path):
    image_name, extension = os.path.splitext(img_path)
    segmented_image_name = image_name + "_marked" + extension  # the future segmented image name to be
    result = subprocess.check_output(['python', os.path.join(CURRENT_FOLDER, 'generate_marker.py'), "-s", img_path])
    return segmented_image_name


def random_forest_predictor(current_data):
    filename = 'randomforestclassl.sav'
    model = pickle.load(open(filename, "rb"))
    predictions = model.predict(current_data)
    return predictions


def xgboost_predictior(current_data):
    model = pickle.load(open("xgboost_better.dat", "rb"))
    predictions_classes = model.predict(current_data)
    return predictions_classes
