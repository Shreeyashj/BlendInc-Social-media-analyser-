import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# Function to preprocess data
def preprocess_data(df):
    df['Sent At'] = pd.to_datetime(df['Sent At'])
    df['DayOfWeek'] = df['Sent At'].dt.day_name()
    return df

# Function to perform exploratory data analysis
def exploratory_data_analysis(df):
    st.subheader("Basic Data Overview")
    st.write(df.head())
    
    # Date analysis
    st.subheader("Invitation Trends Over Time")
    date_counts = df['Sent At'].dt.date.value_counts()
    plt.figure(figsize=(10,4))
    plt.plot(date_counts)
    plt.xticks(rotation=45)
    plt.ylabel('Number of Invitations')
    st.pyplot(plt)

    # Weekday analysis
    st.subheader("Invitation Trends by Weekday")
    weekday_counts = df['DayOfWeek'].value_counts()
    st.bar_chart(weekday_counts)

# Dummy Machine Learning Model (As an example)
def machine_learning_model(df):
    st.subheader("Predictive Model Example")
    st.write("This is a hypothetical model predicting the likelihood of an invitation being accepted.")

    # Encoding categorical data and splitting dataset for demonstration
    df['Direction'] = df['Direction'].apply(lambda x: 1 if x == 'OUTGOING' else 0)
    X = df[['Direction']]  # Simplified for demonstration
    y = np.random.randint(2, size=len(df))  # Dummy response variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and performance
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Model Accuracy: {accuracy:.2f}")

# Streamlit App Layout
def main():
    st.title("LinkedIn Invitations Analysis")
    uploaded_file = st.file_uploader("Upload your invitations.csv", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df)
        
        exploratory_data_analysis(df)
        machine_learning_model(df)

if __name__ == "__main__":
    main()
