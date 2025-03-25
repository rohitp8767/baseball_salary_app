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
import scikit-learn



# Load the model
model = pickle.load(open('baseball.pkl', 'rb'))

# App title
st.title("Baseball Salary Prediction")

# Input form
st.subheader("Enter Player Features:")

# You said 12 important features are used. Let's assume some example ones:
feature_names = ['CAtBat', 'CHits', 'CRuns', 'CRBI', 'AtBat', 'CWalks', 'Hits', 'RBI',
       'Walks', 'CHmRun', 'Runs']

# Collect inputs from user
inputs = []
for name in feature_names:
    val = st.number_input(f"{name}", value=0.0)
    inputs.append(val)

# Prediction button
if st.button("Predict Salary"):
    final_features = [np.array(inputs)]
    prediction = model.predict(final_features)
    output = round(np.exp(prediction[0]), 4)
    st.success(f"Predicted Salary: ₹ {math.floor(output)}")
