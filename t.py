import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import base64
import io

# Load the saved model and dataset
@st.cache_resource
def load_titanic_model():
    return load_model('titanic')

@st.cache_data
def load_dataset():
    return pd.read_csv('titanic.csv')

model = load_titanic_model()
data = load_dataset()

# File download functions
def download_dataset():
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

def download_model():
    with open('titanic.pkl', 'rb') as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    href = f'data:file/pkl;base64,{b64}'
    return href

# Create a function to get user input in main area
def get_user_input():
    st.header("Passenger Information Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric inputs
        age = st.number_input('Age', min_value=0.0, max_value=100.0, value=30.0, step=0.5)
        sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
        parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
        fare = st.number_input('Fare Paid', min_value=0.0, max_value=600.0, value=30.0, step=0.1)
    
    with col2:
        # Categorical inputs
        sex = st.selectbox('Sex', ['Male', 'Female'])
        pclass = st.selectbox('Passenger Class', [1, 2, 3])
        embarked = st.selectbox('Port of Embarkation', 
                               ['Cherbourg', 'Queenstown', 'Southampton'])
    
    # Convert categorical inputs to numerical values
    sex = 1 if sex == 'Female' else 0
    embarked_mapping = {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}
    embarked = embarked_mapping[embarked]
    
    # Store a dictionary into a dataframe
    user_data = {
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Sex': sex,
        'Embarked': embarked
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

def main():
    # Title
    st.title('Titanic Survival Prediction App')
    st.write("""
    This app predicts the likelihood of survival on the Titanic based on passenger information.
    Please fill out the form below and click the 'Predict' button.
    """)
    
    # Sidebar options
    st.sidebar.title("Options")
    
    # View Dataset button
    if st.sidebar.button("View Dataset"):
        st.subheader("Titanic Passenger Dataset")
        st.write(data)
    
    # Download Dataset button
    dataset_download = download_dataset()
    st.sidebar.download_button(
        label="Download Dataset",
        data=data.to_csv(index=False),
        file_name='titanic_dataset.csv',
        mime='text/csv'
    )
    
    # Download Model button
    with open('titanic.pkl', 'rb') as f:
        model_bytes = f.read()
    st.sidebar.download_button(
        label="Download Model",
        data=model_bytes,
        file_name='titanic_model.pkl',
        mime='application/octet-stream'
    )
    
    # Get user input in main area
    user_input = get_user_input()
    
    # Display user input
    st.subheader('Passenger Input Summary')
    st.write(user_input)
    
    # Prediction button
    if st.button('Predict Survival Probability'):
        # Make prediction
        prediction = predict_model(model, data=user_input)
        
        # Display prediction
        st.subheader('Prediction Result')
        prediction_value = prediction['prediction_label'][0]
        prediction_score = prediction['prediction_score'][0]
        
        if prediction_value == 1:
            st.success(f'**Predicted to Survive** (Probability: {prediction_score:.2%})')
            st.info('Higher chance of survival based on passenger characteristics.')
        else:
            st.error(f'**Predicted Not to Survive** (Probability: {1 - prediction_score:.2%})')
            st.warning('Lower chance of survival based on passenger characteristics.')
            
        # Show prediction details expander
        with st.expander("Show detailed prediction metrics"):
            st.write(prediction)

if __name__ == '__main__':
    main()
