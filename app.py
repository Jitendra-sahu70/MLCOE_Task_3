import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack

st.title("Social Media Sentiment Predictor")

# Load model and preprocessing information
model = joblib.load("random_forest_model.pkl")
info = joblib.load("preprocessing.pkl")

encoder = info["encoder"]
numeric_features = info["numeric_features"]
categorical_features = info["categorical_features"]

st.write("Enter the details of a social media post:")

# Categorical inputs
cat_data = {}

for i, feature in enumerate(categorical_features):
    values = encoder.categories_[i]

    cat_data[feature] = st.selectbox(feature,values)

# Numerical inputs
num_data = {}

for feature in numeric_features:
    num_data[feature] = st.number_input(feature,value=0.0)

# Prediction
if st.button("Predict"):

    num_df = pd.DataFrame([num_data])
    cat_df = pd.DataFrame([cat_data])

    cat_encoded = encoder.transform(cat_df)

    final_data = hstack([num_df.values,cat_encoded])

    result = model.predict(final_data)[0]

    st.success("Predicted Sentiment: " + str(result))