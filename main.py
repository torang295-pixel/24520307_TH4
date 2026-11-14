import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import assignment1 as a1
import assignment2 as a2
import assignment3 as a3
import assignment4 as a4

datapath = "chest_xray"
label = {
    'NORMAL': -1,
    'PNEUMONIA': 1
}

def format_data(data: str = 'train'):
    data_path = os.path.join(datapath, data)
    images = []
    labels = []
    for label_name in os.listdir(data_path):
        label_path = os.path.join(data_path, label_name)
        for image_name in tqdm(os.listdir(label_path), desc=f"Processing {label_name}"):
            image_path = os.path.join(label_path, image_name)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            image = cv.resize(image, (256, 256)) if image is not None else None
            image = image.astype(np.float64) / 255.0 if image is not None else None
            image = image.reshape(-1, ) if image is not None else None
            if image is not None:
                images.append(image)
                labels.append(label[label_name])
    return np.array(images), np.array(labels)

X_train, y_train = format_data('train')
X_val, y_val = format_data('val')
X_test, y_test = format_data('test')

for name, model_class, kwargs in (
    ("Assignment 1", a1.SVM, {"epochs": 5}),
    ("Assignment 2", a2.SVM, {"C": 0.1, "epochs": 5}),
    ("Assignment 3", a3.SVM, {"C": 0.1, "epochs": 5}),
    ("Assignment 4", a4.SVM, {"C": 0.1, "max_iter": 100}),
):
    print(f"=== {name} ===")
    svm = model_class(**kwargs)
    svm.fit(X_train, y_train)
    for split, X, y in (
        ("Training", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ):
        score = svm.evaluate(X, y)
        print(f"{split} Accuracy: {score[0]:.4f}, Precision: {score[1]:.4f}, Recall: {score[2]:.4f}, F1 Score: {score[3]:.4f}")

