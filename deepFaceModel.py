from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt


def verify(pairSet):
    # Initialize a dictionary to keep track of model scores
    # model_name : [correct, incorrect]
    model = "VGG-Face"      # The DeepFace library has a variety of models. We chose to use VGG-Face
    model_score = [0, 0]
    misclassified_faces = []

    # Check each pair against each model
    for pair in pairSet:
        # Establish the folder and two images to look at
        path = "archive/lfw-deepfunneled/lfw-deepfunneled/" + pair[0] + "/"
        img1_path = path + pair[1] + ".jpg"
        img2_path = path + pair[2] + ".jpg"
        try:
            res = DeepFace.verify(img1_path, img2_path, model_name = model)
            if res['verified']:
                model_score[0] += 1
            else:
                model_score[1] += 1
                misclassified_faces.append(pair[0])
        # Some faces cannot be identified as a face by the model
        except Exception: 
            model_score[1] += 1
            misclassified_faces.append(pair[0])

    # Display the number of correctly and incorrectly classified faces, as well as who was misclassified
    print(model_score)
    print(misclassified_faces)


def getPeople():
    # Open and read the file
    pairs = open("archive/pairs.csv", "r")
    pairs = pairs.read().splitlines()

    # Extract the pairs and transform to file stubs
    pairs = [pair.split(",")[:3] for pair in pairs][1:5701]
    pairs = [[pair[0], pair[0] + "_" + pair[1].zfill(4), pair[0] + "_" + pair[2].zfill(4)] for pair in pairs]

    return pairs


verify(getPeople())