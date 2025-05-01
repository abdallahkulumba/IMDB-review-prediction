# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.keras')

# Step 2: Helper Functions
# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Step 3: Streamlit App UI
st.title('🎭 IMDB Movie Review Sentiment Analysis')
st.header('🔍 Predict whether a movie review is **Positive** or **Negative**')

# Explanation Text
st.write("""
This app analyzes movie reviews using a **Recurrent Neural Network (RNN)** trained on the IMDB dataset.  
It classifies reviews as **Positive** or **Negative** and visualizes the confidence level of predictions.  
RNNs are great for sentiment analysis because they consider word order and context.
""")

# User input for movie name
movie_name = st.text_input('🎬 Enter the Movie Name')

# User input for movie review
user_review = st.text_area('✍️ Enter Your Review')

if st.button('Classify'):
    if user_review.strip():
        preprocessed_input = preprocess_text(user_review)

        # Make prediction
        prediction = model.predict(preprocessed_input)[0][0]
        sentiment = '😊 Positive' if prediction >= 0.5 else '😞 Negative'

        # Display results
        st.subheader(f'🎬 Movie: **{movie_name}**')
        st.subheader(f'📢 Sentiment: **{sentiment}**')
        st.subheader(f'📊 Prediction Confidence: {prediction:.2f}')
        #plot_confidence_pie(prediction)
    else:
        st.warning('⚠️ Please enter a valid movie review.')


# Sidebar: About the Author Section with Circular Picture & Social Media Links
st.sidebar.subheader("📌 About the Author")

# Display Profile Picture (Crop it manually before uploading)
st.sidebar.image("https://avatars.githubusercontent.com/u/198412563?v=4", caption="Sserujja Abdallah Kulumba", width=150)

st.sidebar.write("""
**Affiliation:** Islamic University of Technology  
**Email:** [abdallahkulumba@iut-dhaka.edu](mailto:abdallahkulumba@iut-dhaka.edu)  
""")

# Social Media Links with Logos
st.sidebar.markdown("""
🔗 **Connect with me**  

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abdallahkulumba)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdallah-kulumba-sserujja/)  
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://www.facebook.com/abdallah.ed.ak)  
""")