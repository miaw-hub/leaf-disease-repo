# Leaf Disease Detection and Treatment System 
This project uses Computer Vision and Machine Learning to identify 21 different types of plant diseases (across Apple, Corn, Potato, Tomato, etc.) and provides instant treatment recommendations.

--Project Highlights--
Dataset: Processed 6,294 images across 21 categories.
Feature Engineering: Unlike simple pixel-based models, this project uses a "fingerprinting" approach extracting:
Color Features: HSV Histograms.
Texture Features: Local Binary Patterns (LBP).
Shape Features: Histogram of Oriented Gradients (HOG).
Top Performance: The Support Vector Machine (SVM) model achieved an accuracy of 89.04%.

--Tech Stack--
Language: Python
Libraries: Scikit-Learn (SVM, KNN, DT), OpenCV (Image processing), Scikit-Image (Feature extraction).

--Categorization & Recommendations--
The system classifies diseases including:
Apple: Black Rot, Cedar Rust, Scab.
Potato: Early Blight, Late Blight.
Tomato: Bacterial Spot, Early/Late Blight.
And more (21 total categories).
After prediction, the system fetches a treatment from the TREATMENTS dictionary (e.g., Fungicide recommendations or pruning tips).

--How to Use--
Clone the repo: git clone https://github.com/your-username/your-repo-name.git
Install requirements: pip install -r requirements.txt
Run the notebook: Open leaf_disease.ipynb and run the cells. You can upload your own leaf image in the final cell to get a prediction and recommendation.
