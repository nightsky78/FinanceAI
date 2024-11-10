# FinanceAI

FinanceAI is a machine learning project designed to classify financial transactions into different categories based on various features such as payer, payee, purpose, IBAN, and amount. The project includes scripts for training a neural network model and performing inference on new data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Data Structure](#data-structure)
- [Model Training](#model-training)
- [Inference](#inference)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/FinanceAI.git
    cd FinanceAI
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Model Training

1. Place your training data in an Excel file named `training_data.xlsx` with a sheet named `Einkommen`.
2. Run the `Modelgen.py` script to train the model:
    ```bash
    python Modelgen.py
    ```

### Inference

1. Place your new data in a CSV file named `new_data.csv`.
2. Run the `inference.py` script to perform inference and save the results:
    ```bash
    python inference.py
    ```

## Files

- `Modelgen.py`: Script for training the neural network model.
- `inference.py`: Script for performing inference on new data.
- `training_data.xlsx`: Excel file containing the training data.
- `new_data.csv`: CSV file containing the new data for inference.
- `result.csv`: CSV file containing the inference results with predicted categories.

## Data Structure

### Training Data File (`training_data.xlsx`)

The training data file should be an Excel file with a sheet named `Einkommen`. The first row should contain the column labels. The following columns are required:

- `Zahlungspflichtige*r`: The payer.
- `Zahlungsempfänger*in`: The payee.
- `Verwendungszweck`: The purpose of the transaction.
- `IBAN`: The IBAN of the account.
- `Betrag`: The amount of the transaction.
- `Kategorie`: The category of the transaction (target variable).

### New Data File (`new_data.csv`)

The new data file should be a CSV file with the following columns:

- `Zahlungspflichtige*r`: The payer.
- `Zahlungsempfänger*in`: The payee.
- `Verwendungszweck`: The purpose of the transaction.
- `IBAN`: The IBAN of the account.
- `Betrag`: The amount of the transaction.

## Model Training

The `Modelgen.py` script performs the following steps:

1. Reads the training data from `training_data.xlsx`.
2. Preprocesses the data by replacing NaN values, vectorizing text columns, and standardizing features.
3. Encodes the target variable `Kategorie` using label encoding.
4. Splits the data into training and testing sets.
5. Defines and trains a neural network model for classification.
6. Saves the trained model, vectorizers, scaler, and label encoder for later use.
7. Evaluates the model on the test data and prints the accuracy and classification report.

## Inference

The `inference.py` script performs the following steps:

1. Loads the saved model, vectorizers, scaler, and label encoder.
2. Reads the new data from `new_data.csv`.
3. Preprocesses the new data in the same way as the training data.
4. Makes predictions using the trained model.
5. Converts the predicted class indices to labels using the label encoder.
6. Adds the predicted labels to the original data.
7. Saves the result to `result.csv`.
8. Prints the result rows with original data and predicted labels.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.