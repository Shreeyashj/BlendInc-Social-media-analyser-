import streamlit as st
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def load_json(file):
    """
    Load JSON file.
    """
    return json.load(file)

def process_data(data):
    """
    Extract and process timestamps and names from the data.
    """
    timestamps = []
    names = []
    for item in data:
        timestamps.append(item['string_list_data'][0]['timestamp'])
        names.append(item['string_list_data'][0]['value'])
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    return dates, names

def analyze_growth(dates):
    """
    Analyze growth based on the dates.
    """
    dates.sort()
    followers_count = list(range(1, len(dates) + 1))
    return pd.DataFrame({'Date': dates, 'Followers': followers_count})

def plot_growth(df):
    """
    Plot the growth of followers over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Followers'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Number of Followers')
    plt.title('Followers Growth Over Time')
    plt.grid(True)
    st.pyplot(plt)

def calculate_growth_rate(df):
    """
    Calculate the rate of growth of followers.
    """
    time_diff = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    growth_rate = (df['Followers'].iloc[-1] - df['Followers'].iloc[0]) / time_diff
    return growth_rate

def predict_followers(df):
    """
    Predict future followers using linear regression.
    """
    df['Date_Ordinal'] = df['Date'].apply(lambda x: x.toordinal())
    X = df[['Date_Ordinal']]
    y = df['Followers']

    model = LinearRegression()
    model.fit(X, y)

    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date, periods=365)
    future_dates_ordinal = [d.toordinal() for d in future_dates]

    predictions = model.predict(np.array(future_dates_ordinal).reshape(-1, 1))
    return future_dates, predictions

# Streamlit interface
st.title('Instagram Followers Growth Analysis')

# File uploader for followers JSON file
followers_file = st.file_uploader("Upload Followers JSON File", type="json")
if followers_file is not None:
    # Load and process the data
    followers_data = load_json(followers_file)
    dates, names = process_data(followers_data)
    df = analyze_growth(dates)

    # Displaying follower names with a styled header
    st.markdown('<h3 style="color: #0066ff; text-decoration: underline; animation: fadeInUp;">List of Followers</h3>', unsafe_allow_html=True)
    st.write(names)

    # Display the count and date range of followers with styled text and animations
    num_followers = len(names)
    st.markdown(f'<p style="font-size: 16px; color: #333; animation: fadeInUp;">Total Number of Followers Received: <span style="font-weight: bold;">{num_followers}</span></p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size: 16px; color: #333; animation: fadeInUp;">Date Range: <span style="font-weight: bold;">{min(dates).strftime("%Y-%m-%d")} to {max(dates).strftime("%Y-%m-%d")}</span></p>', unsafe_allow_html=True)

    # Display followers growth data in a styled table with animations
    st.markdown('<h3 style="color: #0066ff; text-decoration: underline; animation: fadeInUp;">Followers Growth Data</h3>', unsafe_allow_html=True)
    st.dataframe(df.style.set_properties(**{'background-color': '#f5f5f5', 'color': '#333', 'font-size': '16px', 'animation': 'fadeInUp'}))

    # Plotting the followers growth chart with a styled header and animations
    st.markdown('<h3 style="color: #0066ff; text-decoration: underline; animation: fadeInUp;">Followers Growth Chart</h3>', unsafe_allow_html=True)
    plot_growth(df)

    # Calculating and displaying the rate of growth with styled text and animations
    growth_rate = calculate_growth_rate(df)
    st.markdown(f'<p style="font-size: 16px; color: #333; animation: fadeInUp;">Average Rate of Growth: <span style="font-weight: bold;">{growth_rate:.2f}</span> followers per day</p>', unsafe_allow_html=True)

    # Predicting followers growth for the next year with a styled header and animations
    st.markdown('<h3 style="color: #0066ff; text-decoration: underline; animation: fadeInUp;">Predicted Followers Growth for the Next Year</h3>', unsafe_allow_html=True)
    future_dates, predictions = predict_followers(df)
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Followers': predictions})

    # Display the predicted growth chart with animations
    st.line_chart(pred_df.set_index('Date'))

    # Displaying prediction details with styled text and animations
    predicted_growth_rate = (predictions[-1] - df['Followers'].iloc[-1]) / 365
    st.markdown(f'<p style="font-size: 16px; color: #333; animation: fadeInUp;">Predicted Growth Rate: <span style="font-weight: bold;">{predicted_growth_rate:.2f}</span> followers per day</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size: 16px; color: #333; animation: fadeInUp;">Predicted Total Followers in 1 Year: <span style="font-weight: bold;">{int(predictions[-1])}</span></p>', unsafe_allow_html=True)

# Function to process the JSON file
def process_json(uploaded_file):
    json_data = json.load(uploaded_file)
    likes_data = json_data['likes_media_likes']

    # Extracting timestamps and converting them to dates
    dates = [datetime.fromtimestamp(item['string_list_data'][0]['timestamp']).date() for item in likes_data]
    
    return dates

# Function to create visualizations
def create_visualizations(dates):
    df = pd.DataFrame(dates, columns=['Date'])
    df['Count'] = 1
    df = df.groupby('Date').count()
    
    # Bar plot
    fig, ax = plt.subplots()
    ax.bar(df.index, df['Count'])
    ax.set_title("Likes Per Day (Bar Plot)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Likes")
    st.pyplot(fig)
    
    # Line graph
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Count'])
    ax.set_title("Likes Trend Over Time (Line Graph)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Likes")
    st.pyplot(fig)

    return df

# Streamlit App
def main():
    st.title('Instagram Likes Analysis')

    # File upload
    uploaded_file = st.file_uploader("Upload your Instagram JSON file", type="json")
    if uploaded_file is not None:
        # Process JSON file
        dates = process_json(uploaded_file)

        # Visualizations
        df = create_visualizations(dates)

        # Display statistics
        total_likes = df['Count'].sum()
        start_date = df.index.min()
        end_date = df.index.max()
        days = (end_date - start_date).days + 1
        average_likes = total_likes / days
        html_file = open("index.html", "r")
        st.markdown(html_file.read(), unsafe_allow_html=True)
        html_file.close()    
        st.write(f"Total Liked Posts: {total_likes}")
        st.write(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        st.write(f"Daily Average Likes: {average_likes:.2f}")

if __name__ == "__main__":
    main()


