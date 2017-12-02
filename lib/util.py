from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import sys
import tempfile
import cv2
import csv
import glob
import pandas as pd
import numpy as np

IMAGE_DIR = r'C:\Users\crisyp\Documents\machine_learning\Tennis\Tennis_notebook\processing_v2'

def get_list_file_annotations(folder):
    return glob.glob(folder + '/*.annote');


def get_image_training_data(img_tuple):
    """ take the image tuple"""
    img = cv2.imread(img_tuple[0], 1) # i
    (window_x, window_y) = img_tuple[1]
    windows = img_tuple[2]
    data = np.ones((len(windows), window_x * window_y * 3))
    ground_truth = np.zeros((len(windows), 2));
    i = 0
    for window in windows:
        x_pos = int(window[0])
        y_pos = int(window[1])
        is_player = int(window[2]) == 1
        window_image = img[y_pos:y_pos + window_y, x_pos:x_pos + window_x]
        data[i] = window_image.reshape(window_x * window_y * 3)
        if is_player:
            ground_truth[i][0] = 1
            ground_truth[i][1] = 0
        else:
            ground_truth[i][0] = 0
            ground_truth[i][1] = 1
        i = i + 1
    return (data, ground_truth)

def get_data():
    # list of annotation files

    files = get_list_file_annotations(IMAGE_DIR)
    
    training = []
    for file in files:
        # create a tuple representation of annotation
        with open(file, 'r') as f:
            reader = csv.reader(f)
            fileName = IMAGE_DIR + "/" + next(reader)[0]
            dims = next(reader)
            window_x = int(dims[0])
            window_y = int(dims[1])
            windows = []
            for row in reader:
                windows.append(row)
            # now
            (data, ground_truth) = get_image_training_data(
                (fileName, (window_x, window_y), windows)
            )
            for i in range(0, len(data)):
                training.append((data[i], ground_truth[i]))
    training_images = np.zeros((len(training), 40000 * 3))
    training_ground_truth = np.zeros((len(training), 2))

    for i in range(0, len(training)):
        training_images[i] = training[i][0]
        training_ground_truth[i] = training[i][1]
    return (training_images, training_ground_truth)

def get_rescaled_image(h,w):
    (images, ground_truth) = get_data()
    images_down_scaled = np.zeros((len(images), h * w * 3))

    for i in range(0, images.shape[0]):
        img = images[i].reshape(200,200,3).astype(np.uint8)
        res = cv2.resize(img,(h, w), interpolation = cv2.INTER_CUBIC)
        images_down_scaled[i] = res.reshape(h*w*3)
    return (images_down_scaled, ground_truth)
    
    

