# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import joblib

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Load Model, Scaler, and Encoder
with open('model/decision_tree_model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open('model/knnmodel1.pkl', 'rb') as model_file:
    knnmodel = pickle.load(model_file)
    
# with open('model/standard_scaler.pkl', 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)
scaler = joblib.load('model/standard_scaler1.pkl')
# with open('model/onehot_encoder.pkl', 'rb') as encoder_file:
#     encoder = pickle.load(encoder_file)
encoder = joblib.load('model/onehot_encoder.pkl')

# Initialize FastAPI
app = FastAPI()

# CORS Middleware for Angular Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with Angular URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Input Schema
class InputData(BaseModel):
    age: float
    workclass: str
    # fnlwgt: float
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

# Prediction Endpoint
@app.post('/predict')
def predict(data: InputData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([{
        'age': data.age,
        'workclass': data.workclass,
        # 'fnlwgt': data.fnlwgt,
        'education': data.education,
        'marital-status': data.marital_status,
        'occupation': data.occupation,
        'relationship': data.relationship,
        'race': data.race,
        'sex': data.sex,
        'capital-gain': data.capital_gain,
        'capital-loss': data.capital_loss,
        'hours-per-week': data.hours_per_week,
        'native-country': data.native_country
    }])

    # Columns
    numerical_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']

    # Scale Numerical Columns
    try:
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    except AttributeError:
        return {"error": "Scaler is not correctly loaded. Please check the scaler.pkl file."}

    # Encode Categorical Columns
    try:
        encoded_data = encoder.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    except AttributeError as e:
        return {"error": f"Encoder Error: {str(e)}"}
    except UserWarning as e:
        return {"warning": f"Unknown Categories: {str(e)}"}

    # Combine Numerical and Encoded Categorical Data
    final_input = pd.concat([input_data[numerical_cols], encoded_df], axis=1)

    # Prediction
    prediction = model.predict(final_input)
    result = ">50K" if prediction[0] == 1 else "<=50K"
    return {"income": result}


@app.post('/predictknn')
def predict(data: InputData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([{
        'age': data.age,
        'workclass': data.workclass,
        # 'fnlwgt': data.fnlwgt,
        'education': data.education,
        'marital-status': data.marital_status,
        'occupation': data.occupation,
        'relationship': data.relationship,
        'race': data.race,
        'sex': data.sex,
        'capital-gain': data.capital_gain,
        'capital-loss': data.capital_loss,
        'hours-per-week': data.hours_per_week,
        'native-country': data.native_country
    }])

    # Columns
    numerical_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']

    # Scale Numerical Columns
    try:
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    except AttributeError:
        return {"error": "Scaler is not correctly loaded. Please check the scaler.pkl file."}

    # Encode Categorical Columns
    try:
        encoded_data = encoder.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    except AttributeError as e:
        return {"error": f"Encoder Error: {str(e)}"}
    except UserWarning as e:
        return {"warning": f"Unknown Categories: {str(e)}"}

    # Combine Numerical and Encoded Categorical Data
    final_input = pd.concat([input_data[numerical_cols], encoded_df], axis=1)

    # Prediction
    prediction = knnmodel.predict(final_input)
    print(prediction)
    result = ">50K" if prediction[0] == 1 else "<=50K"
    return {"income": result}
