import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib

# Replace 'your_file.xlsx' with the path to your Excel file
xls = pd.ExcelFile('Vermoegensaufbau.xlsx')
sheet_names = xls.sheet_names

# Read the specific sheet into a DataFrame
df_einkommenbalance = pd.read_excel(xls, sheet_name='Einkommen', header=0)

# Display the DataFrame to verify the data
print(df_einkommenbalance.columns)
print(df_einkommenbalance.head())

# List of text columns to be processed
text_columns = ['Zahlungspflichtige*r', 'Zahlungsempfänger*in', 'Verwendungszweck', 'IBAN', 'Betrag']

# Replace NaN values with 0
df_einkommenbalance = df_einkommenbalance.fillna(0)

# Process each text column with TfidfVectorizer
vectorizers = {}
for text_column in text_columns:
    if text_column in df_einkommenbalance.columns:
        vectorizer = TfidfVectorizer()
        text_features = vectorizer.fit_transform(df_einkommenbalance[text_column].astype(str)).toarray()
        text_feature_names = [f"{text_column}_{name}" for name in vectorizer.get_feature_names_out()]
        text_features_df = pd.DataFrame(text_features, columns=text_feature_names)
        df_einkommenbalance = df_einkommenbalance.drop(columns=[text_column])
        df_einkommenbalance = pd.concat([df_einkommenbalance, text_features_df], axis=1)
        vectorizers[text_column] = vectorizer

# Save the vectorizers
joblib.dump(vectorizers, 'vectorizers.pkl')

# Ensure the 'Kategorie' column contains only strings
df_einkommenbalance['Kategorie'] = df_einkommenbalance['Kategorie'].astype(str)

# Convert the target variable 'Kategorie' to numeric using label encoding
label_encoder = LabelEncoder()
df_einkommenbalance['Kategorie'] = label_encoder.fit_transform(df_einkommenbalance['Kategorie'])

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Split the data into features and target
X = df_einkommenbalance.drop(columns=['Kategorie'])
y = df_einkommenbalance['Kategorie']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Define the neural network model for classification
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Use softmax for classification

# Compile the model with a reduced learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training data
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Save the model
model.save('model.h5')

# Evaluate the model using the test data
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Accuracy: {accuracy}')

# Generate the classification report
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_, labels=np.arange(len(label_encoder.classes_))))

