import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("verification.log"),
        logging.StreamHandler()
    ]
)

# Function to preprocess image and return the embeddings
def get_embedding(model, mtcnn, img_path):
    logging.info(f"Processing image: {img_path}")
    img = Image.open(img_path)
    img_cropped = mtcnn(img)
    if img_cropped is None:
        raise ValueError("Face not detected")
    with torch.no_grad():
        embedding = model(img_cropped.unsqueeze(0))
    return embedding

# Function to generate pairs of images from the training folder
def generate_pairs(training_dir):
    logging.info("Generating pairs from training directory")
    pairs = []
    for person_dir in os.listdir(training_dir):
        person_path = os.path.join(training_dir, person_dir)
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, img) for img in os.listdir(person_path) if img.endswith('.jpg')]
            if 2 <= len(images) <= 11:
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        pairs.append((images[i], images[j]))
    logging.info(f"Generated {len(pairs)} pairs")
    return pairs

# Function to verify if two images are of the same person
def verify(pairs):
    logging.info("Starting verification process")
    model_score = [0, 0]
    misclassified_faces = []

    # Load the FaceNet model
    model = InceptionResnetV1(pretrained='vggface2').eval()
    mtcnn = MTCNN(image_size=160, margin=0)

    def process_pair(img1_path, img2_path):
        try:
            embedding1 = get_embedding(model, mtcnn, img1_path)
            embedding2 = get_embedding(model, mtcnn, img2_path)
            distance = torch.dist(embedding1, embedding2).item()
            threshold = 1.0
            if distance < threshold:
                logging.info(f"Match: {img1_path} and {img2_path}")
                return (True, img1_path, img2_path)
            else:
                logging.info(f"No Match: {img1_path} and {img2_path}")
                return (False, img1_path, img2_path)
        except Exception as e:
            logging.error(f"Error processing pair ({img1_path}, {img2_path}): {e}")
            return (False, img1_path, img2_path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_pair, img1, img2) for img1, img2 in pairs]
        for future in as_completed(futures):
            result, img1, img2 = future.result()
            if result:
                model_score[0] += 1
            else:
                model_score[1] += 1
                misclassified_faces.append((img1, img2))

    logging.info(f"Model Score: {model_score}")
    logging.info(f"Misclassified Faces: {misclassified_faces}")

# Set the directories relative to the project directory
training_dir = os.path.join(os.getcwd(), 'training')

logging.info("Script started")
pairs = generate_pairs(training_dir)
verify(pairs)
logging.info("Script finished")
