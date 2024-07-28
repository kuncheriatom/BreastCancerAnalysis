# Breast Cancer Diagnosis Prediction

This project involves building and deploying a machine learning model to predict breast cancer diagnosis based on various features derived from digitized images of fine needle aspirates (FNA) of breast masses. The model uses an Artificial Neural Network (ANN) and is integrated into a web application built with Streamlit.

## Project Components

1. **Model**: An ANN model trained on the Breast Cancer Wisconsin (Diagnostic) dataset.
2. **Feature Selector**: A feature selector (using `sklearn.feature_selection.SelectKBest`) to select the most important features for the model.
3. **Scaler**: A `StandardScaler` used to normalize the input features.
4. **Streamlit App**: A web application that allows users to input feature values and get predictions.

## Dataset

The dataset used for training and testing the model is the Breast Cancer Wisconsin (Diagnostic) dataset. It contains features computed from images of cell nuclei.

## Installation

To run this project locally, follow these steps:

### 1. **Clone the Repository**

   git clone https://github.com/kuncheriatom/BreastCancerAnalysis
   cd BreastCancerAnalysis


### 2. Create and Activate a Virtual Environment


python -m venv env

#On Windows use `env\Scripts\activate` to activate


### 3. Install Dependencies


pip install -r requirements.txt


### 4. Download Pre-trained Model and Artifacts

Ensure the following files are in the project directory:

- `ann_model_with_dropout.h5` (The trained ANN model)
- `kbest.pkl` (The pickled feature selector)
- `scaler.pkl` (The pickled scaler)

If these files are not present, you may need to train the model and save these artifacts.

## Usage

### 1. Start the Streamlit App


streamlit run app.py


### 2. Navigate to the Web App

Open your web browser and go to `http://localhost:8501`.

### 3. Input Data and Get Prediction

Enter the values for the features in the provided input fields. The application will display the predicted diagnosis as either 'Malignant' or 'Benign'.

## Feature Descriptions

The Streamlit app provides descriptions for each feature used in the model. The features are:

- `radius_mean`: Mean distance from the center to points on the perimeter of the cell nucleus
- `perimeter_mean`: Mean perimeter length of the cell nucleus
- `area_mean`: Mean area of the cell nucleus
- `smoothness_mean`: Local variation in radius lengths of the cell nucleus
- `compactness_mean`: Perimeter^2 / area - 1.0
- `concavity_mean`: Severity of concave portions of the contour
- `concave_points_mean`: Number of concave portions of the contour
- `symmetry_mean`: Symmetry of the cell nucleus
- `fractal_dimension_mean`: Coastline approximation - 1
- `radius_worst`: Largest mean radius of the cell nucleus
- `perimeter_worst`: Largest mean perimeter of the cell nucleus
- `area_worst`: Largest mean area of the cell nucleus
- `concavity_worst`: Largest mean severity of concave portions of the contour
- `concave_points_worst`: Largest mean number of concave portions of the contour

## Notes

- Ensure you have the required version of Python and the necessary packages listed in `requirements.txt`.
- The Streamlit app should be run in an environment where the model and other artifacts are accessible.



## Acknowledgements

- The dataset used in this project is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

