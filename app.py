import streamlit as st
import joblib

# Load your model and any necessary preprocessing steps
model = joblib.load('email_classification_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define custom CSS for styling
custom_css = """
<style>
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #333333;
        text-align: center;
        margin-bottom: 30px;
    }
    .text-input {
        font-size: 18px;
        color: #555555;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #cccccc;
        margin-bottom: 20px;
        width: 80%;
        box-sizing: border-box;
    }
    .button {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
        background-color: #4CAF50;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .button:hover {
        background-color: #45a049;
    }
    .result {
        font-size: 24px;
        font-weight: bold;
        color: #333333;
        margin-top: 20px;
    }
</style>
"""

# Inject custom CSS into Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Define the Streamlit UI
def main():
    st.markdown("<h1 class='title'>Support Mail Category Prediction</h1>", unsafe_allow_html=True)
    
    subject_input = st.text_input('Enter email subject:', key='subject_input')
    body_input = st.text_input('Enter email body:', key='body_input')  

    if st.button('Predict', key='predict_button'): 
        # Perform predictions
        if body_input:
            # Preprocess the user input
            user_input_features = vectorizer.transform([body_input])
            # Make prediction
            prediction = model.predict(user_input_features)
            # Display result
            st.markdown(f"<p class='result'>Predicted class: <span style='color: #4CAF50;'>{prediction[0]}</span></p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

