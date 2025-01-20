import joblib

encoder = joblib.load('model/onehot_encoder.pkl')
print(type(encoder))  # Should print: <class 'sklearn.preprocessing._encoders.OneHotEncoder'>
print(encoder.get_feature_names_out())  # Verify feature names
