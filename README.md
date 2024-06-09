# Support-Mail-BOT-Prediction
## Problem Statement:
Design an Algorithm that will perform data analytics on a given email data and categorize email data belonging to specific category and predict the relevance. The algorithm should also filter out the spams in addition to the category that has been filtered out earlier. Some of the Categorization can be Resumes, Application, Meetings, HR, Spam, etc.

## Solution Overview

The email classification model in this repository leverages natural language processing (NLP) techniques and machine learning algorithms to automatically categorize emails. The model preprocesses the text data by removing punctuation, converting text to lowercase, tokenizing, removing stopwords, and lemmatizing words to extract meaningful features. It then uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert the preprocessed text into numerical features. Finally, a Logistic Regression classifier is trained on the vectorized features to predict the category of each email.


Dataset: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset


## Introduction

This repository contains a machine learning model for classifying emails into predefined categories such as Resumes, Applications, Meetings, HR, and Spam. The model uses text preprocessing techniques and TF-IDF for feature extraction, followed by a Logistic Regression classifier to perform the classification.
- Resumes
- Applications
- Meetings
- HR
- Spam

## Tech Stack Used

- **Python**: The core programming language used for developing the email classification model.
- **Natural Language Toolkit (NLTK)**: Python library for natural language processing tasks such as tokenization, stopwords removal, and lemmatization.
- **scikit-learn**: Python library for machine learning tasks such as model training, evaluation, and feature extraction.
- **matplotlib** and **seaborn**: Python libraries for data visualization, used to visualize the model's performance metrics and data distribution.
- **Jupyter Notebook**: Interactive computing environment used for data exploration, experimentation, and development of the email classification model.

## Key Features

- **Text Preprocessing**: The model preprocesses the text data to clean and standardize it before vectorization.
- **TF-IDF Vectorization**: TF-IDF is used to convert the preprocessed text data into numerical features while capturing the importance of each word in the corpus.
- **Logistic Regression Classifier**: A Logistic Regression model is trained on the vectorized features to classify emails into predefined categories.
- **Evaluation Metrics**: The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score to assess its effectiveness in email classification.
- **Visualization**: The repository includes visualizations of class distribution, top TF-IDF features for each category, confusion matrix, and model accuracy to provide insights into the model's performance and decision-making process.


## Data Preprocessing

The text data is preprocessed using the following steps:
- Remove punctuation and convert text to lowercase.
- Tokenize the text.
- Remove stopwords.
- Lemmatize the words.

## Final Results

After training and evaluating the Logistic Regression model on the email classification dataset, the following results were obtained:

## Accuracy
Testing Accuracy: 0.9003
