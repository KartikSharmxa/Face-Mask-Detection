

#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    # Path to the downloaded dataset
    data_directory = 'dataset/'

    # Load the dataset
    X, y = load_dataset(data_directory)

    # Check if there are at least two classes present
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("Error: The dataset must contain at least two classes for SVM classification.")
        return

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Extract HOG features from the images
    X_train_features = extract_features(X_train)
    X_val_features = extract_features(X_val)

    # Create and train the SVM model
    svm_model = SVC(kernel='rbf', gamma='scale')
    svm_model.fit(X_train_features, y_train)

    # Evaluate the model
    train_accuracy = evaluate_model(svm_model, X_train_features, y_train)
    val_accuracy = evaluate_model(svm_model, X_val_features, y_val)

    print("Training Accuracy:", train_accuracy)
    print("Validation Accuracy:", val_accuracy)

    # Prediction
    predict_from_user_input(svm_model)


def load_dataset(data_directory):
    X = []
    y = []

    for class_name in os.listdir(data_directory):
        class_directory = os.path.join(data_directory, class_name)
        if os.path.isdir(class_directory):
            # Assign class labels based on folder names
            if class_name == 'with_mask':
                class_label = 1  # Class label for 'with_mask'
            elif class_name == 'without_mask':
                class_label = 0  # Class label for 'without_mask'
            else:
                continue  # Skip folders that are not 'with_mask' or 'without_mask'

            for filename in os.listdir(class_directory):
                img_path = os.path.join(class_directory, filename)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (128, 128))
                X.append(img_resized)
                y.append(class_label)

    return np.array(X), np.array(y)



def extract_features(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
        features.append(hog_features)
    return np.array(features)


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy


def predict_from_user_input(model):
    while True:
        input_image_path = input("Enter the path to the image you want to predict (or 'quit' to exit): ")

        if input_image_path.lower() == 'quit':
            print("Exiting the program.")
            break

        try:
            input_image = cv2.imread(input_image_path)
            if input_image is not None:
                input_image_resized = cv2.resize(input_image, (128, 128))
                input_features = extract_features([input_image_resized])

                # Making prediction
                prediction = model.predict(input_features)

                if prediction[0] == 1:
                    print('The person in the image is wearing a mask')
                else:
                    print('The person in the image is not wearing a mask')
            else:
                print("Invalid image path. Please check the path and try again.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()


