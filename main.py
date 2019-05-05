import os
import numpy as np
import pandas as pd
import cv2
import pickle
from matplotlib import pyplot as plt

DB_FILE = 'features.bin'
IMAGE_FOLDER = './images'

MATCH_RATIO = 0.8

def loadDB():
    dbfile = open(DB_FILE, "rb")
    images = pickle.load(dbfile)

    imageFeatures = []
    for imagePath, features in images:
        _keypoints = []
        _descriptors = []
        for kp in features:
            _kp = cv2.KeyPoint(x=kp[0][0],y=kp[0][1], _size=kp[1], _angle=kp[2],
                              _response=kp[3], _octave=kp[4], _class_id=kp[5])
            _keypoints.append(_kp)
            
            desc = kp[-1]
            _descriptors.append(desc)
        obj = {
            "kp":_keypoints,
            "desc":np.array(_descriptors)
        }
        imageFeatures.append((imagePath,obj))
    return imageFeatures


def extract(imagePath):
    cvImg = cv2.imread(imagePath)
    # Convert image to grayscale
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
    # Initialize sift detector
    sift = cv2.xfeatures2d.SIFT_create()
    # Get SIFT Features
    keypoints, descriptors = sift.detectAndCompute(cvImg, None)
    return (keypoints,descriptors)

def matchFeatures(kp1,kp2,desc1,desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)
    print("Features found: ", len(matches))

    distances = np.array([[m1.distance, m2.distance] for m1, m2 in matches])
    ratios = distances[:, 0] / distances[:, 1]
    good_features = np.argwhere(ratios < MATCH_RATIO).ravel()

    x = np.array(matches)[good_features]
    p1 = [m[0].queryIdx for m in x]
    p1 = np.array(p1, dtype=int)
    p2 = [m[0].trainIdx for m in x]
    p2 = np.array(p2, dtype=int)

    kp1 = np.array(kp1)
    kp2 = np.array(kp2)

    p1 = np.array([k.pt for k in kp1[p1]])
    p2 = np.array([k.pt for k in kp2[p2]])

    # Calculate Homography
    homography, s = cv2.findHomography(p2, p1, cv2.RANSAC, 5.0)
    print(s.sum() / s.size)

if __name__ == "__main__":
    print(f"Loading image features from {DB_FILE}")
    imageFeatures = loadDB()
    print(f"Loaded {len(imageFeatures)} from db")
    idx1 = 4
    idx2 = 6
    print(imageFeatures[idx1][0],imageFeatures[idx2][0])
    kp1, desc1 = imageFeatures[idx1][1]['kp'], imageFeatures[idx1][1]['desc']
    kp2, desc2 = imageFeatures[idx2][1]['kp'], imageFeatures[idx2][1]['desc']
    matchFeatures(kp1,kp2,desc1,desc2)
