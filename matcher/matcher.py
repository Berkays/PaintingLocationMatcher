import os
import argparse
from pathlib import Path
import pickle
import cv2
import numpy as np

from pymongo import MongoClient,ASCENDING
from pymongo.errors import BulkWriteError
from bson.binary import Binary

DB_URL = "localhost:27017"
DB_NAME = "paintingLocationDB"
DB_COLLECTION = "paintingLocations"
TRAIN_IMAGE_FOLDER = '../extractor/train'
TEST_IMAGE_FOLDER = 'test'
FLOOR_PLAN_IMAGE = 'floorPlan.jpg'

RATIO_TEST = 0.80 # Ratio for choosing good features
MATCH_RATIO = 0.66 # Ratio needed to match 2 images.

def loadDB():
    client = MongoClient(DB_URL)
    db = client[DB_NAME]
    collection = db[DB_COLLECTION]
    
    images = list(collection.find())

    imageFeatures = []
    for image in images:
        features = pickle.loads(image['features'])
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
        imageFeatures.append((image['path'], image['location'], obj))
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

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    distances = np.array([[m1.distance, m2.distance] for m1, m2 in matches])
    ratios = distances[:, 0] / distances[:, 1]
    good_features = np.argwhere(ratios < RATIO_TEST).ravel()

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
    _, s = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    accuracy = s.sum() / s.size

    return accuracy,good

def showPlan(location):
    img = cv2.imread(FLOOR_PLAN_IMAGE)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, location[:-2],
                (300, 30),
                font,
                1.2,
                (0, 0, 0))

    point = (350, 100)
    if(location[-1] == "b"):
        point = (500,100)

    textPoint = (point[0] - 60, point[1] + 30)
    cv2.circle(img, point, 8, (0, 0, 255), -1)
    cv2.putText(img, 'You are here',
                textPoint,
                font,
                1.2,
                (0, 0, 255))

    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEF University Location Finder with images")
    parser.add_argument('-u', '--url', type=str, default=DB_URL,help=f'MongoDB Url ({DB_URL})')
    parser.add_argument('-d', '--dir', type=str,default=TRAIN_IMAGE_FOLDER, help=f'Directory containing images ({TRAIN_IMAGE_FOLDER})')
    parser.add_argument('-r', '--ratio', type=float, default=RATIO_TEST,
                        help=f'Ratio test to filter good features ({RATIO_TEST})')
    parser.add_argument('-m', '--match', type=float, default=MATCH_RATIO,
                        help=f'Minimum match accuracy to select ({MATCH_RATIO})')
    parser.add_argument("file", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    DB_URL = args.url
    IMAGE_FOLDER = args.dir
    MATCH_RATIO = args.match
    RATIO_TEST = args.ratio

    if(len(args.file) == 0):
        print("No test image specified.")
        os._exit()
    else:
        test_file = args.file[0]
        print("Test File: ",args.file[0])

    print("Loading image features from ",DB_URL)
    imageFeatures = loadDB()
    imageFeaturesCount = len(imageFeatures)
    print("Loaded ", imageFeaturesCount)

    kp1,desc1 = extract(test_file)

    bestAccuracy = 0 
    bestMatch = None
    goodFeatures = None

    for i in range(0, imageFeaturesCount):
        kp2, desc2 = imageFeatures[i][2]['kp'], imageFeatures[i][2]['desc']
        accuracy,_goodFeatures = matchFeatures(kp1,kp2,desc1,desc2)
        print("Accuracy: ",accuracy)
        if(accuracy >= MATCH_RATIO and accuracy > bestAccuracy):
            bestAccuracy = accuracy
            bestMatch = imageFeatures[i]
            goodFeatures = _goodFeatures

    print()
    if(bestMatch is None):
        print("No match found")
    else:
        print("Search: ",  test_file)
        print("Best Match: ", bestMatch[0])
        print("Accuracy: ", bestAccuracy)
        print("Location: ", bestMatch[1])

        img1 = cv2.imread(test_file)
        img2 = cv2.imread(os.path.join(TRAIN_IMAGE_FOLDER, bestMatch[0]))
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, bestMatch[2]['kp'], goodFeatures, None, flags=2)
        img3 = cv2.resize(img3,None,fx=0.5,fy=0.5)
        cv2.imshow('match',img3)
        showPlan(bestMatch[1])
