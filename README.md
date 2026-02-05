# Leaf Disease Detection and Treatment System 
This project uses Computer Vision and Machine Learning to identify 21 different types of plant diseases (across Apple, Corn, Potato, Tomato, etc.) and provides instant treatment recommendations. 
note: Optimized for Google Colab; change the DATA_DIR path if running locally.

--**Project Highlights**--
Dataset: Processed 6,294 images across 21 categories.
Feature Engineering: Unlike simple pixel-based models, this project uses a "fingerprinting" approach extracting:
Color Features: HSV Histograms.
Texture Features: Local Binary Patterns (LBP).
Shape Features: Histogram of Oriented Gradients (HOG).
Top Performance: The Support Vector Machine (SVM) model achieved an accuracy of 89.04%.

--**Tech Stack**--
Language: Python
Libraries: Scikit-Learn (SVM, KNN, DT), OpenCV (Image processing), Scikit-Image (Feature extraction).

--**Data Processing Pipeline**--
The project follows a structured Computer Vision pipeline using OpenCV and OS:
Data Loading (os): Systematically walked through directory structures to label images based on folder names.
Preprocessing (cv2): * Images resized to a uniform dimension.
Color conversion from BGR to HSV for better color-based disease detection.
Feature Extraction: * HOG: To capture the shape of leaf lesions.
LBP: To capture the texture of the disease (e.g., crusty vs. smooth spots).
Color Histograms: To detect yellowing or browning.
Classification: Comparing SVM, KNN, and Decision Trees to find the most accurate model.

--**Categorization & Recommendations**--
The system classifies diseases including:
Apple: Black Rot, Cedar Rust, Scab.
Potato: Early Blight, Late Blight.
Tomato: Bacterial Spot, Early/Late Blight.
And more (21 total categories).
After prediction, the system fetches a treatment from the TREATMENTS dictionary (e.g., Fungicide recommendations or pruning tips).


--**How to Use**--
You don't need to re-train the model!
Ensure leaf_model_pack.pkl and scaler.pkl are in the main directory.
Open the notebook and run the "Load Model" cell.
Provide the path to a leaf image from the sample_images/ folder to see the diagnosis and treatment instantly.
or You can run the prediction directly from your terminal:
--> python predict.py
Then, simply enter the path to any image when prompted.
