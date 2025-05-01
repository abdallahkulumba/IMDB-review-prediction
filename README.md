# Simple RNN IMDB review Project

This project implements a Simple Recurrent Neural Network (RNN) for sentiment analysis on the IMDB movie reviews dataset. The goal is to classify movie reviews as positive or negative based on the text content.

## Overview

The Simple RNN model is designed to process sequences of text data, making it suitable for tasks like sentiment analysis. This project utilizes the IMDB dataset, which contains 50,000 movie reviews labeled as either positive or negative.

## Requirements

To run this project, you need to install the following dependencies:

- **TensorFlow 2.15.0**: A powerful library for building and training machine learning models.
- **Pandas**: A data manipulation library that provides data structures for efficiently handling structured data.
- **NumPy**: A library for numerical computations in Python.
- **Scikit-learn**: A machine learning library that provides tools for model evaluation and preprocessing.
- **TensorBoard**: A visualization tool for monitoring training progress and model performance.
- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python.
- **Streamlit**: A framework for building interactive web applications for machine learning and data science projects.
- **Scikeras**: A wrapper for Keras that integrates with Scikit-learn.

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

To run the Streamlit app, use the following command in your terminal:

```bash
streamlit run your_app_file.py
```

Replace `your_app_file.py` with the name of your Streamlit app file in mycase the name was 'main.py'. This will start a local server, and you can access the app in your web browser at `http://localhost:8501`.

## Usage Instructions

1. **Data Preparation**: Ensure that the IMDB dataset is available and properly formatted for the model.
2. **Model Training**: Train the Simple RNN model using the provided training scripts. Monitor the training process using TensorBoard.
3. **Running the App**: After training, run the Streamlit app to interact with the model. You can input movie reviews and receive predictions on their sentiment.
4. **Visualizations**: The app may include visualizations of model performance, such as accuracy and loss graphs.

## Conclusion

This project serves as a practical implementation of a Simple RNN for sentiment analysis. It demonstrates the process of building, training, and deploying a machine learning model using Streamlit for user interaction.

Feel free to explore the code and modify it to suit your needs!
