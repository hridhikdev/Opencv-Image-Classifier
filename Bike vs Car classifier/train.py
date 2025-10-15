import cv2
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    h = hog.compute(image)
    return h.flatten()

def load_images_and_labels(base_path):
    features = []
    labels = []
    classes = ['car', 'bike']
    
    for label, class_name in enumerate(classes):
        class_folder = os.path.join(base_path, class_name)
        for filename in os.listdir(class_folder):
            filepath = os.path.join(class_folder, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 128)) 
            hog_feat = extract_hog_features(img)
            features.append(hog_feat)
            labels.append(label)
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    dataset_path = 'dataset'  
    print("Loading images and extracting HOG features...")
    features, labels = load_images_and_labels(dataset_path)

    print(f"Extracted features from {len(features)} images")

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print("Training Linear SVM classifier...")
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    print("Evaluating on test data...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['car', 'bike']))
    joblib.dump(clf, 'bike_car_svm_model.pkl')
    print("Model saved as bike_car_svm_model.pkl")
