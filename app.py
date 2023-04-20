import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

# Set page title
st.set_page_config(page_title='ML Model Execution App')

# Set app title
st.title('ML Model Execution App')

# Create sidebar for selecting model type
model_type = st.sidebar.selectbox('Select Model Type', ['Regression', 'Classification'])

# Create file uploader for CSV file
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

# Read CSV file and display data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    # Create sidebar for selecting target variable
    target_variable = st.sidebar.selectbox('Select Target Variable', df.columns)

    # Split data into training and testing sets
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create machine learning model based on selected model type
    if model_type == 'Regression':
        model = LinearRegression()
    else:
        model = LogisticRegression()

    # Fit the model on training data and display accuracy score
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    st.write('Accuracy:', accuracy)