import cv2
import numpy as np
import pickle
import os
from skimage.feature import hog, local_binary_pattern

# 1. Feature Extraction Logic (Must be identical to your training code)
def extract_advanced_features(img):
    # Resize to the same size used during training
    img_resized = cv2.resize(img, (128, 128))
    
    # Color Features (HSV Histogram)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(hist, hist).flatten()
    
    # Texture Features (LBP)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_feat = lbp_hist.astype("float")
    lbp_feat /= (lbp_feat.sum() + 1e-6)
    
    # Shape Features (HOG)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), visualize=False)
    
    # Combine all features
    return np.hstack([color_feat, lbp_feat, hog_feat])

# 2. Loading the Model and Scaler
def load_system():
    if not os.path.exists('leaf_model_pack.pkl') or not os.path.exists('scaler.pkl'):
        print("Error: .pkl files not found! Please ensure they are in the same folder.")
        return None, None, None, None

    with open('leaf_model_pack.pkl', 'rb') as f:
        data = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    return data['model'], data['categories'], data['treatments'], scaler

# 3. The Main Execution Logic
def main():
    print("--- Leaf Disease Predictor ---")
    model, categories, treatments, scaler = load_system()
    
    if model is None: return

    image_path = input("Enter the path to the leaf image (e.g., sample.jpg): ")
    
    if not os.path.exists(image_path):
        print("File not found.")
        return

    # Process Image
    img = cv2.imread(image_path)
    features = extract_advanced_features(img).reshape(1, -1)
    scaled_features = scaler.transform(features)
    
    # Predict
    prediction_index = model.predict(scaled_features)[0]
    result = categories[prediction_index]
    recommendation = treatments.get(result, "Consult an expert.")

    print("\n" + "="*30)
    print(f"IDENTIFIED: {result}")
    print(f"TREATMENT:  {recommendation}")
    print("="*30)

if __name__ == "__main__":
    main()
