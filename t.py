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

# File download functions
def download_dataset():
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return b64

def get_user_input():
    st.header("Passenger Information Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric inputs
        pclass = st.selectbox('Passenger Class', [1, 2, 3])
        age = st.number_input('Age', min_value=0.0, max_value=100.0, value=30.0, step=0.5)
        sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
        parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
    
    with col2:
        # Categorical inputs
        sex = st.selectbox('Sex', ['Male', 'Female'])
        fare = st.number_input('Fare Paid', min_value=0.0, max_value=600.0, value=30.0, step=0.1)
        embarked = st.selectbox('Port of Embarkation', ['Cherbourg', 'Queenstown', 'Southampton'])
    
    # Convert categorical inputs to numerical values
    sex = 1 if sex == 'Female' else 0
    embarked_mapping = {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}
    embarked = embarked_mapping[embarked]
    
    # Create dataframe with all expected columns in correct order
    user_data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    
    # Create a DataFrame with all columns the model expects
    features = pd.DataFrame([user_data])
    
    # Ensure all columns are present (add missing with default values if needed)
    expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for col in expected_columns:
        if col not in features.columns:
            features[col] = 0  # or appropriate default value
    
    return features[expected_columns]  # Return columns in correct order

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
    st.write(user_input)
    
    if st.button('Predict Survival Probability'):
        try:
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
