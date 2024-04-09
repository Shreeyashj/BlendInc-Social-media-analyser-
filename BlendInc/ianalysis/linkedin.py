import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# Function to verify if the uploaded file is correct
def verify_file(data):
    required_columns = {'First Name', 'Last Name', 'Connected On'}
    return required_columns.issubset(data.columns)

# Function to get categories of companies and job titles
def get_categories(data):
    companies = data['Company'].dropna().unique().tolist()
    job_titles = data['Position'].dropna().unique().tolist()
    return companies, job_titles

# Function to analyze growth and plot the graph
def analyze_growth(data):
    data['Connected On'] = pd.to_datetime(data['Connected On'])
    data.sort_values(by='Connected On', inplace=True)
    data['Cumulative Connections'] = np.arange(1, len(data) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(data['Connected On'], data['Cumulative Connections'], color='blue', marker='o')
    plt.title('Growth of LinkedIn Connections Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Connections')
    plt.grid(True)
    plt.tight_layout()
    return plt

# Function to predict future connections
def predict_future_connections(data, future_months):
    data['Connected On'] = pd.to_datetime(data['Connected On'])
    data['Days Since First Connection'] = (data['Connected On'] - data['Connected On'].min()).dt.days

    X = data[['Days Since First Connection']]
    y = data['Cumulative Connections']

    model = LinearRegression()
    model.fit(X, y)

    current_day = (data['Connected On'].max() - data['Connected On'].min()).days
    future_days = current_day + np.array(future_months) * 30  # Approximating a month as 30 days
    future_predictions = model.predict(future_days.reshape(-1, 1))
    
    return dict(zip(future_months, future_predictions.astype(int)))

# Function to plot bar graphs for companies and job titles
def plot_distribution(data, column, title):
    counts = data[column].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='skyblue')
    plt.title(f'Top 10 {title}')
    plt.ylabel('Number of Connections')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# Streamlit UI
st.title("LinkedIn Connections Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your 'Connections.csv' file", type='csv')
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, skiprows=3)
    
    if verify_file(data):
        # Display total number of connections
        st.write(f"Total Connections: {data.shape[0]}")
        
        # Display categories
        companies, job_titles = get_categories(data)
        category_type = st.selectbox("Select category type:", ["Company", "Job Title"])

        if category_type == "Company":
            selected_company = st.selectbox("Select a Company", companies)
            company_connections = data[data['Company'] == selected_company]
            st.write("Connections at " + selected_company + ":")
            st.write(company_connections[['First Name', 'Last Name']].reset_index(drop=True))
        else:
            selected_title = st.selectbox("Select a Job Title", job_titles)
            title_connections = data[data['Position'] == selected_title]
            st.write("Connections with title " + selected_title + ":")
            st.write(title_connections[['First Name', 'Last Name']].reset_index(drop=True))

        # Plot growth analysis
        st.pyplot(analyze_growth(data))

        # Plot distributions
        plot_distribution(data, 'Company', 'Companies in Connections')
        plot_distribution(data, 'Position', 'Job Titles in Connections')

        # Future predictions
        prediction_months = [3, 6, 12]
        predictions = predict_future_connections(data, prediction_months)
        for months, prediction in predictions.items():
            st.write(f"Predicted connections in the next {months} months: {prediction}")

    else:
        st.error("This doesn't seem to be the right file. Please upload the correct 'Connections.csv' file.")
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
