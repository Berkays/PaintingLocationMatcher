import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import pickle

from pymongo import MongoClient,ASCENDING
from pymongo.errors import BulkWriteError
from bson.binary import Binary

DB_URL = "database:27017"
DB_NAME = "paintingLocationDB"
DB_COLLECTION = "paintingLocations"
IMAGE_FOLDER = 'train'

# Serializes keypoints and descriptors extracted from SIFT Detector.
def serializeFeatures(keypoints,descriptors):
    serialized = []
    for kp,desc in zip(keypoints,descriptors):
        obj = (kp.pt,kp.size,kp.angle,kp.response,kp.octave,kp.class_id,desc)
        serialized.append(obj)
    return serialized

# Extract SIFT Features and build image database object
def extract(imagePath):
    cvImg = cv2.imread(imagePath)
    # Convert image to grayscale
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
    # Initialize sift detector
    sift = cv2.xfeatures2d.SIFT_create()
    # Extract SIFT Features from image
    keypoints, descriptors = sift.detectAndCompute(cvImg, None)
    features = serializeFeatures(keypoints,descriptors)
    
    imagePath = os.path.relpath(imagePath,IMAGE_FOLDER)
    location = Path(imagePath).parent.name

    return {
        'path':imagePath,
        'location':location,
        'features': Binary(pickle.dumps(features))
    }

# Extract and serialize sift features then insert into the MongoDB
def buildDB():
    processedImages = []
    locations = os.listdir(IMAGE_FOLDER)
    if(len(locations) == 0):
        print("No location directories found.")
        return

    for location in locations:
        images = os.listdir(os.path.join(IMAGE_FOLDER, location))
        for image in images:
            imagePath = os.path.join(IMAGE_FOLDER,location,image)
            print("Processing: ",image)
            processedImages.append(extract(imagePath))

    if(len(processedImages) == 0):
        print("No images found to process.")
        return

    print(f"{len(processedImages)} Images processed.")
    try:
        client = MongoClient(DB_URL)
        db = client[DB_NAME]
        collection = db[DB_COLLECTION]
        collection.create_index([('path', ASCENDING)],unique=True)
        status = collection.insert_many(processedImages, ordered=False)
        print(len(status.inserted_ids),"Features saved to DB.")
    except BulkWriteError as bwe:
        print(bwe.details['nInserted'],"Features saved to DB.")
        print("Some duplicate elements are not inserted.")
    finally:
        client.close()

# Purge MongoDB database
def dropDB():
    try:
        client = MongoClient(DB_URL)
        client.drop_database(DB_NAME)
    finally:
        client.close()
        print("Database deleted succesfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEF University Location Finder DB Builder")
    parser.add_argument('-p', '--purge', type=int, default=0, help='Purge database')
    parser.add_argument('-u', '--url', type=str, default=DB_URL, help='MongoDB Url (localhost:27017)')
    parser.add_argument('-d', '--dir', type=str,default=IMAGE_FOLDER, help='Directory containing images')

    args = parser.parse_args()
    
    DB_URL = args.url
    IMAGE_FOLDER = args.dir

    if(args.purge == 1):
        dropDB()

    buildDB()
