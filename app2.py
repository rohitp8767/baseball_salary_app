import streamlit as st
import numpy as np
import pickle
import mysql.connector



import base64



def insert_to_mysql(data):
    conn = mysql.connector.connect(
        # host='localhost',       # or 127.0.0.1
        # user='root',   # e.g., root
        # password='123456789',
        # database='baseball'

        host='sql12.freesqldatabase.com',
        # Database name: sql12774116
        user='sql12774116',
        password='HDpEzDwiXH',
        # Port number: 3306
        database='sql12774116'


    )
    
    cursor = conn.cursor()

    insert_query = '''
        INSERT INTO player_inputs (
            AtBat, CWalks, CAtBat, HmRun, CHits, CRuns, CRBI, RBI, PredictedSalary
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''

    cursor.execute(insert_query, data)
    conn.commit()
    conn.close()

# Load the trained model
with open('baseball_ada.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Baseball Player Salary Predictor 💰")


def set_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image filename
set_bg_from_local("baseball-bg.jpg")

st.markdown("### Enter Player Stats Below:")

# Define input fields for the top 8 important features
CWalks = st.number_input('Career Walks', min_value=0)
CAtBat = st.number_input('Career AtBat', min_value=0)
HmRun = st.number_input('Home Runs', min_value=0)
CHits = st.number_input('Career Hits', min_value=0)
CRuns = st.number_input('Career Runs', min_value=0)
CRBI = st.number_input('Career RBI', min_value=0)
AtBat = st.number_input('AtBat', min_value=0)
RBI = st.number_input('RBI', min_value=0)

# Predict button
if st.button('Predict Salary'):

    # Input data in the correct order (top 8 features, no categorical ones)
    input_data = np.array([[CWalks, CAtBat, HmRun, CHits, CRuns, CRBI, AtBat, RBI]])
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    actual_salary = np.exp(prediction)  # Reversing the log transformation on salary
    st.success(f'Estimated Salary: ${actual_salary:,.2f}')

    # Insert the prediction and inputs into MySQL
    insert_to_mysql((
        AtBat, CWalks, CAtBat, HmRun, CHits, CRuns, CRBI, RBI, round(actual_salary, 2)
    ))
