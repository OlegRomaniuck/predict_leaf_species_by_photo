import numpy as np
import cv2
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
import json
import itertools
import os
import os.path
import ctypes
from DIPLOMA.my_dipl.kursovaya.ImageFeature import *

sqlite_file = './SQLite Database/leavesDatabase.sqlite'
BASIC_PATH = os.path.join( os.path.dirname(os.path.abspath(__file__)), "data_set")
data_set_folder = ["train", "val"]
def arrayStringEncode(originalArray):
    encodedString = ""
    for item in originalArray:
        encodedString = encodedString + str(item) + ','

    encodedString = encodedString[:-1]
    return encodedString


def stringArrayDecode(encodedString):
    recovArray = []

    histList = encodedString.split(',')
    for item in histList:
        recovArray.append(float(item))

    return recovArray

def loadTreeData():
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()

    qry = open('./SQLite Database/createTable.sql', 'r').read()
    sqlite3.complete_statement(qry)
    try:
        cursor.executescript(qry)
    except Exception as e:
        raise e

    qry = open('./SQLite Database/insertTreeData.sql', 'r').read()
    sqlite3.complete_statement(qry)
    try:
        cursor.executescript(qry)
    except Exception as e:
        raise e

    conn.commit()
    cursor.close()
    conn.close()

def loadProcessed():  # Load parameters for leaf images in database
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    df = pd.read_sql_query("SELECT TreeID,Paths FROM TreeData", conn)  # Obtain all Tree entries in TreeData Table
    count = 0
    items = []

    for index, row in df.iterrows():  # Iterate through all leaf entries
        leaf_folders = str(row["Paths"]).split(', ')
        for lf_folder in leaf_folders:
            for fld in data_set_folder:
                path_to_leaf = os.path.join(BASIC_PATH, fld,lf_folder)
                print(path_to_leaf)
                for image_name in os.listdir(path_to_leaf):
                    image_path = os.path.join(path_to_leaf, image_name)
                    print("Try got get information from {}".format(image_path))
                    item = []
                    treeid = row['TreeID']
                    leafid = "{}_{}_{}".format(lf_folder,fld, image_name)
                    try:
                        length, width, area, perimeter, aspect_ratio, form_factor, rectangularity, hu, hist = getAllFeatures(image_path)
                    except BaseException:
                        print("Cannot handle with picture\n !!!!!! {}".format(image_path))
                        continue
                    item.append(treeid)
                    item.append(leafid)
                    item.append(length)
                    item.append(width)
                    item.append(area)
                    item.append(perimeter)
                    item.append(aspect_ratio)
                    item.append(form_factor)
                    item.append(rectangularity)
                    item.append(arrayStringEncode(hu))
                    item.append(arrayStringEncode(hist))
                    items.append(item)
                    print(item)
                    try:
                        cursor.execute('INSERT INTO Processed VALUES(?,?,?,?,?,?,?,?,?,?,?)', item)
                    except Exception as e:
                        raise e
                    conn.commit()
    print("Processed images")
    conn.commit()
    cursor.close()
    conn.close()
    print("Wrote to database")


def obtainXY():
    conn = sqlite3.connect(sqlite_file)
    df = pd.read_sql_query("SELECT * FROM Processed", conn)

    x1 = df['Length']
    x2 = df['Width']
    x3 = df['Area']
    x4 = df['Perimeter']
    x5 = df['AspectRatio']
    x6 = df['FormFactor']
    x7 = df['Rectangularity']
    x8 = df['Hu'].tolist()
    x9 = df['Hist'].tolist()

    count = 0
    while (count < len(x8)):
        huArray = stringArrayDecode(x8[count])
        x8[count] = huArray
        count = count + 1

    count = 0
    while (count < len(x9)):
        lbpHist = stringArrayDecode(x9[count])
        x9[count] = lbpHist
        count = count + 1

    y = df['TreeID'].tolist()  # np.array(df['treeID'].tolist()).flatten('F')
    x = np.column_stack((x3, x4, x5, x6, x7, x8, x9))  # Convert
    return x, y

def getTreeData(treeId):
    conn = sqlite3.connect(sqlite_file)
    query = "SELECT TreeID,ScientificName,CommonName FROM TreeData WHERE TreeID = " + treeId

    dataFrame = pd.read_sql_query(query, conn)
    dt = dataFrame.set_index('TreeID').T.to_dict()
    jsonResult = json.dumps(dt[int(treeId)])

    print(jsonResult)
    return jsonResult


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    plt.figure()
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')

    plt.show()

