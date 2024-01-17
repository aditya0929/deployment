
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from tempfile import NamedTemporaryFile
import os
import PyPDF2
import re

# Set up SQLite database connection
conn = sqlite3.connect('/content/user_data.db')
cursor = conn.cursor()

# Create a table for user data if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        transaction_amount INTEGER,
        average_expenditure INTEGER,
        comparison_with_avg_expenditure INTEGER,
        transaction_count_7_days INTEGER,
        "Total Credit Amount" INTEGER,
        prediction INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# Load the pre-trained model, scaler, and label encoder
model_data = joblib.load('/content/random_forest_.pkl')
label_encoder = model_data['label_encoder']
scaler = model_data['scaler']
model = model_data['model']
features = model_data['features']

# Function to extract data from PDF
def extract_data_from_pdf(pdf_filename):
    extracted_data = {}

    with open(pdf_filename, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            # Define regular expressions for each attribute
            attribute_patterns = {
                "age": r"age:\s*(\d+)",
                "transaction_amount": r"transaction_amount:\s*([\d.]+)",
                "average_expenditure": r"average_expenditure:\s*([\d.]+)",
                "comparison_with_avg_expenditure": r"comparison_with_avg_expenditure:\s*([\d.]+)",
                "transaction_count_7_days": r"transaction_count_7_days:\s*(\d+)",
                "Total Credit Amount": r"Total Credit Amount:\s*([\d.]+)",
            }

            # Extract values using regular expressions
            for feature, pattern in attribute_patterns.items():
                match = re.search(pattern, text)
                if match:
                    extracted_data[feature] = match.group(1)

    return extracted_data

# Streamlit app code
st.title('Fraud Detection App')

# File upload section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Display user input or uploaded file data
if uploaded_file is not None:
    # Extract data from the uploaded PDF
    temp_pdf = NamedTemporaryFile(delete=False)
    temp_pdf.write(uploaded_file.read())
    temp_pdf.close()
    
    extracted_data = extract_data_from_pdf(temp_pdf.name)

    # Display the extracted data
    st.subheader('Uploaded PDF Data:')
    uploaded_data = pd.DataFrame(extracted_data, index=[0])
    st.table(uploaded_data)

    # Use the extracted data to fill the input space
    user_input = {}
    for feature in features:
        if feature != 'fraud_indicator' and feature in extracted_data:
            user_input[feature] = st.number_input(f'Enter {feature}', step=1, value=int(float(extracted_data[feature])))
        else:
            user_input[feature] = st.number_input(f'Enter {feature}', step=1, value=0)

    # Display user input
    st.subheader('User Input:')
    user_data_input = pd.DataFrame(user_input, index=[0])
    st.table(user_data_input)

# Display user input space if no PDF is uploaded
else:
    st.subheader('Enter Transaction Details:')
    user_input = {}

    for feature in features:
        if feature != 'fraud_indicator':
            user_input[feature] = st.number_input(f'Enter {feature}', step=1, value=0)

    # Display user input
    st.subheader('User Input:')
    user_data_input = pd.DataFrame(user_input, index=[0])
    st.table(user_data_input)

# Make prediction
user_data_processed = user_data_input.copy()  # No need to preprocess for SQLite
user_data_scaled = scaler.transform(user_data_processed)
prediction = model.predict(user_data_scaled)

# Insert user input and prediction into the database
if st.button('Submit'):
    cursor.execute('''
        INSERT INTO user_data (
            age,
            transaction_amount,
            average_expenditure,
            comparison_with_avg_expenditure,
            transaction_count_7_days,
            "Total Credit Amount",
            prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_input['age'],
        user_input['transaction_amount'],
        user_input['average_expenditure'],
        user_input['comparison_with_avg_expenditure'],
        user_input['transaction_count_7_days'],
        user_input['Total Credit Amount'],
        prediction[0]
    ))
    conn.commit()
    st.success('Data submitted successfully!')

    # Display prediction for fraud detection
    st.subheader('Prediction for Fraud Detection:')
    if prediction[0] == 1:
        st.warning('This transaction is flagged as potentially fraudulent!')
    else:
        st.success('This transaction is not flagged as fraudulent!')

# Streamlit app code for visualization on the sidebar
st.sidebar.title('Fraud Detection App - Visualization')

# Display user IDs on the sidebar
user_ids = pd.read_sql_query('SELECT id FROM user_data ORDER BY timestamp DESC LIMIT 5', conn)['id'].tolist()
selected_user_id = st.sidebar.selectbox('Select User ID:', user_ids)

# Retrieve data for the selected user ID
selected_data = pd.read_sql_query(f'SELECT * FROM user_data WHERE id={selected_user_id}', conn)

# Display selected user data on the sidebar
st.sidebar.subheader(f'Selected User Data (User ID: {selected_user_id}):')
st.sidebar.table(selected_data)

# Visualization - Transaction Amount Excess Chart
if selected_data.shape[0] > 0:
    st.sidebar.subheader('Transaction Amount Excess Chart')

    # Threshold value for transaction amount
    transaction_amount_threshold = 110000

    # Excess transactions
    excess_transactions = max(0, selected_data['transaction_amount'].values[0] - transaction_amount_threshold)

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(['Excess Transactions', 'Remaining Transactions'], [excess_transactions, transaction_amount_threshold - excess_transactions], color=['red', 'green'])
    ax.set_xlabel('Number of Transactions')
    ax.set_title('Transaction Amount Excess Chart')
    st.sidebar.pyplot(fig)
    st.sidebar.write(f'Transaction Amount Threshold: {transaction_amount_threshold}')

# Visualization - Comparison with Avg Expenditure Excess Chart
if selected_data.shape[0] > 0:
    st.sidebar.subheader('Comparison with Avg Expenditure Excess Chart')

    # Threshold value for comparison with avg expenditure
    comparison_with_avg_expenditure_threshold = 30000

    # Excess value
    excess_value = max(0, selected_data['comparison_with_avg_expenditure'].values[0] - comparison_with_avg_expenditure_threshold)

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(['Excess Value', 'Remaining Value'], [excess_value, comparison_with_avg_expenditure_threshold - excess_value], color=['red', 'green'])
    ax.set_xlabel('Value')
    ax.set_title('Comparison with Avg Expenditure Excess Chart')
    st.sidebar.pyplot(fig)
    st.sidebar.write(f'Comparison with Avg Expenditure Threshold: {comparison_with_avg_expenditure_threshold}')

# Visualization - Total Credit Amount Excess Chart
if selected_data.shape[0] > 0:
    st.sidebar.subheader('Total Credit Amount Excess Chart')

    # Threshold value for total credit amount
    total_credit_amount_threshold = 150000

    # Excess value
    excess_value_credit = max(0, selected_data['Total Credit Amount'].values[0] - total_credit_amount_threshold)

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(['Excess Value', 'Remaining Value'], [excess_value_credit, total_credit_amount_threshold - excess_value_credit], color=['red', 'green'])
    ax.set_xlabel('Value')
    ax.set_title('Total Credit Amount Excess Chart')
    st.sidebar.pyplot(fig)
    st.sidebar.write(f'Total Credit Amount Threshold: {total_credit_amount_threshold}')

# Close the database connection when done
conn.close()

# Sidebar
st.sidebar.title('Additional Information')
st.sidebar.markdown('This Streamlit app is for demonstration purposes only.')
