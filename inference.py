import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the saved model, vectorizers, scaler, and label encoder
model = load_model('model.h5')
vectorizers = joblib.load('vectorizers.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Print the encoded classes
print("Encoded classes in the label encoder:")
for index, class_label in enumerate(label_encoder.classes_):
    print(f"{index}: {class_label}")

# Load the new data for inference
new_data = pd.read_csv('new_data.csv', delimiter=';')

# Keep a copy of the original data
original_data = new_data.copy()

# Select only the relevant columns
new_data = new_data[['Zahlungspflichtige*r', 'Zahlungsempfänger*in', 'Verwendungszweck', 'IBAN', 'Betrag']]

# Replace NaN values with 0
new_data = new_data.fillna(0)

# Convert 'Betrag' column to float
new_data['Betrag'] = new_data['Betrag'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)

# Process the new data in the same way as the training data
for text_column, vectorizer in vectorizers.items():
    if text_column in new_data.columns:
        text_features = vectorizer.transform(new_data[text_column].astype(str)).toarray()
        text_feature_names = [f"{text_column}_{name}" for name in vectorizer.get_feature_names_out()]
        text_features_df = pd.DataFrame(text_features, columns=text_feature_names)
        new_data = new_data.drop(columns=[text_column])
        new_data = pd.concat([new_data, text_features_df], axis=1)

# Ensure the new data has the same columns as the training data
new_data = new_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Standardize the features
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_scaled)
predicted_classes = np.argmax(predictions, axis=1)

# Convert the predicted class indices to labels using the label encoder
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Add the predicted labels to the original data
original_data.insert(0, 'Predicted Kategorie', predicted_labels)

# Save the result to a CSV file
original_data.to_csv('result.csv', index=False, sep=';')

# Print the result rows with original data and predicted labels
print("Result rows with original data and predicted labels:")
print(original_data)