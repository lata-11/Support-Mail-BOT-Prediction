# Support-Mail-BOT-Prediction
Problem Statement:
Design an Algorithm that will perform data analytics on a given email data and categorize email data belonging to specific category and predict the relevance. The algorithm should also filter out the spams in addition to the category that has been filtered out earlier. Some of the Categorization can be Resumes, Application, Meetings, HR, Spam, etc.

Dataset: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset

This repository contains a machine learning model for classifying emails into predefined categories such as Resumes, Applications, Meetings, HR, and Spam. The model uses text preprocessing techniques and TF-IDF for feature extraction, followed by a Logistic Regression classifier to perform the classification.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Saving the Model](#saving-the-model)
- [License](#license)

## Introduction

This project aims to automatically categorize emails based on their content using natural language processing (NLP) techniques and machine learning. The categories considered are:
- Resumes
- Applications
- Meetings
- HR
- Spam

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/email-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd email-classification
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset with email bodies and their respective categories.
2. Run the provided script to preprocess the data, train the model, evaluate its performance, and visualize the results.

## Data Preprocessing

The text data is preprocessed using the following steps:
- Remove punctuation and convert text to lowercase.
- Tokenize the text.
- Remove stopwords.
- Lemmatize the words.

## Final Results

After training and evaluating the Logistic Regression model on the email classification dataset, the following results were obtained:

### Accuracy
Testing Accuracy: 0.9003
