# import numpy as np
# import pickle
# import math
# from flask import Flask, request, jsonify, render_template

# app = Flask(__name__,template_folder= "template", static_folder= "staticfiles") ## assign Flask = app
# model = pickle.load(open('baseball.pkl','rb'))   ### import model - Random Forest

# @app.route('/')  # define root folder
# def home():
#     return render_template('index.html')  # read index.html file

# @app.route('/predict', methods=['POST'])   ###transfer data from html to python / server

# def predict():
#     int_features = [float(x)  for x in request.form.values()]   # request for data values
#     final_features = [np.array(int_features)]  # convert into array
#     prediction = model.predict(final_features)  # Predict


#     output = round(np.exp(prediction[0]),4)  # to Get original Salary (doing expoenentiation)

#     return render_template('index.html',
#                            prediction_text="Salary {}".format(math.floor(output)))

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)





# ['CAtBat', 'CHits', 'CRuns', 'CRBI', 'AtBat', 'CWalks', 'Hits', 'RBI',
#        'Walks', 'CHmRun', 'Runs']



import streamlit as st
import numpy as np
import pickle
import math

import numpy
import pandas
import sklearn

from PIL import Image
import base64



# # Load the model
# model = pickle.load(open('baseball.pkl', 'rb'))

# # App title
# st.title("Baseball Salary Prediction")

# # Input form
# st.subheader("Enter Player Features:")

# # You said 12 important features are used. Let's assume some example ones:
# feature_names = ['CAtBat', 'CHits', 'CRuns', 'CRBI', 'AtBat', 'CWalks', 'Hits', 'RBI',
#        'Walks', 'CHmRun', 'Runs']

# # Collect inputs from user
# inputs = []
# for name in feature_names:
#     val = st.number_input(f"{name}", value=0.0)
#     inputs.append(val)

# # Prediction button
# if st.button("Predict Salary"):
#     final_features = [np.array(inputs)]
#     prediction = model.predict(final_features)
#     output = round(np.exp(prediction[0]), 4)
#     st.success(f"Predicted Salary: ₹ {math.floor(output)}")







# Load model
with open("baseball.pkl", "rb") as f:
    model = pickle.load(f)

# Set Streamlit page configuration
st.set_page_config(page_title="Baseball Salary Predictor", layout="centered")

# Set background image
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_from_local("baseball-bg.jpg")

# Inject custom CSS
def local_css():
    css = """
    <style>
    h1 {
        color: white;
        text-align: center;
        margin-top: 30px;
    }
    label, .stNumberInput>div>div>input {
        font-weight: bold;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 0.5em 1.5em;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

local_css()

# Title
st.markdown("<h1>Baseball Salary Predictor</h1>", unsafe_allow_html=True)

# Input fields for all 12 features
st.markdown("### Enter Player Stats:")
col1, col2 = st.columns(2)

with col1:
    CAtBat = st.number_input("Career At Bats (CAtBat)", min_value=0)
    CHits = st.number_input("Career Hits (CHits)", min_value=0)
    CRuns = st.number_input("Career Runs (CRuns)", min_value=0)
    CRBI = st.number_input("Career RBIs (CRBI)", min_value=0)
    AtBat = st.number_input("Current At Bats (AtBat)", min_value=0)
    CWalks = st.number_input("Career Walks (CWalks)", min_value=0)

with col2:
    Hits = st.number_input("Current Hits (Hits)", min_value=0)
    RBI = st.number_input("Current RBIs (RBI)", min_value=0)
    Walks = st.number_input("Current Walks (Walks)", min_value=0)
    CHmRun = st.number_input("Career Home Runs (CHmRun)", min_value=0)
    Runs = st.number_input("Current Runs (Runs)", min_value=0)

# Padding the layout to make it even
st.markdown("<br>", unsafe_allow_html=True)

# Predict
if st.button("Predict"):
    features = [[CAtBat, CHits, CRuns, CRBI, AtBat, CWalks,
                 Hits, RBI, Walks, CHmRun, Runs]]
    prediction = model.predict(features)
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")

