import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the ANN model
model = tf.keras.models.load_model('ann_model_with_dropout.h5')

# Load the pickled feature selector
with open('kbest.pkl', 'rb') as f:
    kbest = pickle.load(f)

# Load the scaler used during training
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
def main():
    st.title("Breast Cancer Type Prediction")

    st.write("Enter the values for the following features:")

    # Input fields for the user with help text
    radius_mean = st.text_input("Radius Mean", help="Mean of distances from the center to points on the perimeter of the cell nuclei")
    perimeter_mean = st.text_input("Perimeter Mean", help="Mean value of the perimeter of the cell nuclei")
    area_mean = st.text_input("Area Mean", help="Mean value of the area of the cell nuclei")
    concavity_mean = st.text_input("Concavity Mean", help="Mean severity of concave portions of the cell nuclei contour")
    concave_points_mean = st.text_input("Concave Points Mean", help="Mean number of concave portions of the cell nuclei contour")
    radius_worst = st.text_input("Radius Worst", help="Worst (largest) value for the radius of the cell nuclei")
    perimeter_worst = st.text_input("Perimeter Worst", help="Worst (largest) value for the perimeter of the cell nuclei")
    area_worst = st.text_input("Area Worst", help="Worst (largest) value for the area of the cell nuclei")
    concavity_worst = st.text_input("Concavity Worst", help="Worst (largest) value for the concavity of the cell nuclei")
    concave_points_worst = st.text_input("Concave Points Worst", help="Worst (largest) value for the concave points of the cell nuclei")

    # Button to trigger prediction
    if st.button("Predict"):
        # Function to validate and convert input to float
        def validate_and_convert(value, label):
            try:
                return float(value)
            except ValueError:
                st.error(f"Please enter a valid number for {label}.")
                return None

        # Validate and convert inputs
        radius_mean = validate_and_convert(radius_mean, "Radius Mean")
        perimeter_mean = validate_and_convert(perimeter_mean, "Perimeter Mean")
        area_mean = validate_and_convert(area_mean, "Area Mean")
        concavity_mean = validate_and_convert(concavity_mean, "Concavity Mean")
        concave_points_mean = validate_and_convert(concave_points_mean, "Concave Points Mean")
        radius_worst = validate_and_convert(radius_worst, "Radius Worst")
        perimeter_worst = validate_and_convert(perimeter_worst, "Perimeter Worst")
        area_worst = validate_and_convert(area_worst, "Area Worst")
        concavity_worst = validate_and_convert(concavity_worst, "Concavity Worst")
        concave_points_worst = validate_and_convert(concave_points_worst, "Concave Points Worst")

        # Check if all required inputs are provided and valid
        if None in [radius_mean, perimeter_mean, area_mean, concavity_mean, concave_points_mean, radius_worst, perimeter_worst, area_worst, concavity_worst, concave_points_worst]:
            st.error("Please provide valid values for all features.")
        else:
            # Create DataFrame from user inputs
            input_data = {
                'radius_mean': radius_mean,
                'perimeter_mean': perimeter_mean,
                'area_mean': area_mean,
                'concavity_mean': concavity_mean,
                'concave points_mean': concave_points_mean,
                'radius_worst': radius_worst,
                'perimeter_worst': perimeter_worst,
                'area_worst': area_worst,
                'concavity_worst': concavity_worst,
                'concave points_worst': concave_points_worst,
            }
            input_df = pd.DataFrame([input_data])

            # Ensure the input DataFrame has the same columns as the scaler expects
            input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

            # Scale the Input Data
            scaled_input = scaler.transform(input_df)

            # Apply the Feature Selector to the Scaled Input Data
            input_df_transformed = kbest.transform(scaled_input)

            # Make Predictions
            prediction = model.predict(input_df_transformed)
            prediction_label = 'Malignant' if prediction > 0.5 else 'Benign'
            st.write(f"The predicted diagnosis is: {prediction_label}")

if __name__ == "__main__":
    main()
