import os
import numpy as np
import pandas as pd
import cv2
import pickle

DB_FILE = 'features.bin'
IMAGE_FOLDER = './images'

locations = os.listdir(IMAGE_FOLDER)

def serializeFeatures(keypoints,descriptors):
    serialized = []
    for kp,desc in zip(keypoints,descriptors):
        obj = (kp.pt,kp.size,kp.angle,kp.response,kp.octave,kp.class_id,desc)
        serialized.append(obj)
    return serialized

def extract(imagePath):
    cvImg = cv2.imread(imagePath)
    # Convert image to grayscale
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
    # Initialize sift detector
    sift = cv2.xfeatures2d.SIFT_create()
    # Get SIFT Features
    keypoints, descriptors = sift.detectAndCompute(cvImg, None)
    features = serializeFeatures(keypoints,descriptors)
    return (imagePath,features)

def buildDB():
    db = []
    for location in locations:
        images = os.listdir(os.path.join(IMAGE_FOLDER, location))
        for image in images:
            imagePath = os.path.join(IMAGE_FOLDER,location,image)
            db.append(extract(imagePath))

    print(f"{len(db)} Images processed.")
    dbfile = open(DB_FILE, "wb")
    pickle.dump(db,dbfile,protocol=-1)
    dbfile.close()
    print("Features extracted to ", DB_FILE)

def dropDB():
    if(os._exists(DB_FILE)):
        os.remove(DB_FILE)

if __name__ == "__main__":
    # Get locations from folder structure
    dropDB()
    # Extract SIFT Features from images.
    buildDB()
