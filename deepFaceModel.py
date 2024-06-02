from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img1_path = "img1.jpeg"
img2_path = "img2.jpeg"

def verify(img1_path, img2_path, model_name = "VGG-Face"):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    plt.imshow(img1[:, :,   ::-1])
    plt.show()
    plt.imshow(img2[:, :, ::-1])
    plt.show()
    # result = DeepFace.verify(img1_path, img2_path)
    # result = Deepface.verify(img1_path, img2_path, model_name = VGG-Face)
    result = DeepFace.verify(img1_path, img2_path, model_name = "Facenet")
    print("Result:", result)

    if result['verified']:
        print("They are the same person.")
    else:
        print("They are not the same person.")

verify("img1.jpeg", "img2.jpeg", model_name = "OpenFace")
