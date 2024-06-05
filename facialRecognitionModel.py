from pathlib import Path
import pickle
import face_recognition
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Define paths
DEFAULT_ENCODINGS_PATH = Path("/Users/kkragas/Desktop/CSC 466/466_MOdel1/encodings.pkl")
training = Path("/Users/kkragas/Desktop/CSC 466/466_MOdel1/sample1")

###############################################################################################################

# train data
def encode_known_faces(encodings_location: Path = DEFAULT_ENCODINGS_PATH):
    # stores labeled images and their encodings
    labels = []
    encodings = []

    # iterate over files in the subdirectories of the training directory
    for filepath in Path(training).glob("*/*"): 
        # extracts the name of current subdirectory that is being iterated
        label = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model="hog")
        # transalate each detected face into encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            labels.append(label)
            encodings.append(encoding)

    labeled_encodings = {"labels": labels, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(labeled_encodings, f)

###############################################################################################################

def recognize_faces(image_location: str, encodings_location: Path = DEFAULT_ENCODINGS_PATH):
    # open and load the saved encodings using pickle
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    
    # load in unlabeled image you want to classify
    unlabelled_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(unlabelled_image, model="hog")
    input_face_encodings = face_recognition.face_encodings(unlabelled_image, input_face_locations)

    #iterate through each pair of face location and encoding  at the same time
    for face_location, unknown_encoding in zip(input_face_locations, input_face_encodings):
        # identify the face using helper function
        label = _recognize_face(unknown_encoding, loaded_encodings)
        # if face is not recognized, identity is unknown
        if not label:
            label = "Unknown"

        return(label)

def _recognize_face(unknown_encoding, loaded_encodings):
    # compare unlabeled encoding with the labeled encodings -> boolean value of whether each unknown encoding matches known
    # unknown: [36t4] known: [457q, 877a, 36t4, a123] bool_matches: [FALSE, FALSE, TRUE, FALSE]
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    # pairs each boolean match with the corresponding label then counts the number of matches
    votes = Counter(label for match, label in zip(boolean_matches, loaded_encodings["labels"]) if match)
    # return most commonly occuring label
    if votes:
        return votes.most_common(1)[0][0]


recognize_faces("/Users/kkragas/Desktop/CSC 466/466_MOdel1/sample2/Mickey_Mouse.jpeg")
###############################################################################################################
# predict using the validation set
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

def validate(validationFile: str):
    actual_labels = []
    predicted_labels = []
    for filepath in Path(validationFile).rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in IMAGE_EXTENSIONS:
            actual = filepath.stem
            predicted = recognize_faces(image_location=str(filepath.absolute()))

            if predicted is None:
               predicted = "Unknown"

            actual_labels.append(actual)
            predicted_labels.append(predicted)

    #print(actual_labels)
    #print(predicted_labels)
    
    return actual_labels, predicted_labels

###############################################################################################################
# measure model performance
def check_performance(actual_labels, predicted_labels):
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, average='weighted')
    recall = recall_score(actual_labels, predicted_labels, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

###############################################################################################################
# encode_known_faces()
a, p = validate("/Users/kkragas/Desktop/CSC 466/466_MOdel1/validation")
check_performance(a, p)

#a, p = validate("/Users/kkragas/Desktop/CSC 466/466_MOdel1/sample2")
#check_performance(a, p)
#recognize_faces("/Users/kkragas/Desktop/CSC 466/466_MOdel1/sample2/Mickey_Mouse.jpeg")
#recognize_faces("/Users/kkragas/Desktop/CSC 466/466_MOdel1/sample2/Alec_Baldwin.jpg")

#python 466_FRModel1.py