import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import base64

# Load the saved model and dataset
@st.cache_resource
def load_titanic_model():
    return load_model('titanic')

@st.cache_data
def load_dataset():
    return pd.read_csv('titanic.csv')

try:
    model = load_titanic_model()
    data = load_dataset()
except Exception as e:
    st.error(f"Error loading model or dataset: {str(e)}")
    st.stop()

def get_user_input():
    st.header("Passenger Information Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Main features
        pclass = st.selectbox('Passenger Class', [1, 2, 3])
        age = st.number_input('Age', min_value=0.0, max_value=100.0, value=30.0, step=0.5)
        sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
        parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
    
    with col2:
        # Other features
        sex = st.selectbox('Sex', ['Male', 'Female'])
        fare = st.number_input('Fare Paid', min_value=0.0, max_value=600.0, value=30.0, step=0.1)
        embarked = st.selectbox('Port of Embarkation', ['Cherbourg', 'Queenstown', 'Southampton'])
    
    # Convert categorical inputs
    sex = 1 if sex == 'Female' else 0
    embarked_mapping = {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}
    embarked = embarked_mapping[embarked]
    
    # Create a dictionary with all expected columns
    user_data = {
        'Passengerid': 0,  # Dummy value since we don't collect this
        'Age': age,
        'Fare': fare,
        'Sex': sex,
        'sibsp': sibsp,
        'Parch': parch,
        'Pclass': pclass,
        'Embarked': embarked,
        '2urvived': 0,  # Dummy value for target column
    }
    
    # Add all the zero columns your model expects
    for i in range(19):
        user_data[f'zero.{i}' if i > 0 else 'zero'] = 0
    
    # Create DataFrame with all columns in correct order
    features = pd.DataFrame([user_data])
    
    # Ensure we have all columns the model expects
    expected_columns = [
        'Passengerid', 'Age', 'Fare', 'Sex', 'sibsp', 'zero', 'zero.1', 'zero.2',
        'zero.3', 'zero.4', 'zero.5', 'zero.6', 'zero.7', 'zero.8', 'zero.9',
        'zero.10', 'zero.11', 'zero.12', 'zero.13', 'zero.14', 'zero.15',
        'zero.16', 'zero.17', 'zero.18', 'Parch', 'Pclass', 'Embarked', '2urvived'
    ]
    
    # Add any missing columns with default value 0
    for col in expected_columns:
        if col not in features.columns:
            features[col] = 0
    
    return features[expected_columns]

def main():
    st.title('Titanic Survival Prediction App')
    st.write("""
    This app predicts the likelihood of survival on the Titanic based on passenger information.
    Please fill out the form below and click the 'Predict' button.
    """)
    
    # Sidebar options
    st.sidebar.title("Options")
    
    if st.sidebar.button("View Dataset"):
        st.subheader("Titanic Passenger Dataset")
        st.write(data)
    
    # Download buttons
    st.sidebar.download_button(
        label="Download Dataset",
        data=data.to_csv(index=False),
        file_name='titanic_dataset.csv',
        mime='text/csv'
    )
    
    try:
        with open('titanic.pkl', 'rb') as f:
            st.sidebar.download_button(
                label="Download Model",
                data=f,
                file_name='titanic_model.pkl',
                mime='application/octet-stream'
            )
    except Exception as e:
        st.sidebar.error("Could not load model for download")
    
    # Get user input
    user_input = get_user_input()
    
    st.subheader('Passenger Input Summary')
    # Show only relevant columns in display
    display_cols = ['Pclass', 'Sex', 'Age', 'sibsp', 'Parch', 'Fare', 'Embarked']
    st.write(user_input[display_cols])
    
    if st.button('Predict Survival Probability'):
        try:
            # Make prediction
            prediction = predict_model(model, data=user_input)
            
            st.subheader('Prediction Result')
            prediction_value = prediction['prediction_label'][0]
            prediction_score = prediction['prediction_score'][0]
            
            if prediction_value == 1:
                st.success(f'✅ Predicted to Survive (Probability: {prediction_score:.2%})')
                st.info('Higher chance of survival based on passenger characteristics.')
            else:
                st.error(f'❌ Predicted Not to Survive (Probability: {1 - prediction_score:.2%})')
                st.warning('Lower chance of survival based on passenger characteristics.')
                
            with st.expander("Show detailed prediction metrics"):
                st.write(prediction)
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Please check that all input values are valid and try again.")

if __name__ == '__main__':
    main()
