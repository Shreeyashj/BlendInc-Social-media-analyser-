import streamlit as st
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt

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


