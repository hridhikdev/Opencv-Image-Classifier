import cv2
import joblib

def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    h = hog.compute(image)
    return h.flatten()

def predict_image(image_path, clf):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    feat = extract_hog_features(img)
    pred = clf.predict([feat])
    return 'car' if pred == 0 else 'bike'

if __name__ == "__main__":
    model = joblib.load('bike_car_svm_model.pkl')
    test_image_path = 'test_image.jpg'  # Change this to your test image path

    prediction = predict_image(test_image_path, model)
    print(f"Prediction: {prediction}")
