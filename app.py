from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
from preprocess import preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef
import io
import csv
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = joblib.load('nslkdd_xgb.pkl')

# Class mapping 
attack_categories = {0: 'Brute Force',
                     1: 'Denial of Service (DoS/DDoS and Botnet)',
                     2: 'Normal',
                     3: 'Reconnaissance',
                     4: 'Remote to Local (R2L)',
                     5: 'User to Root (U2R)'}

# Reverse class mapping
attack_categories_reverse = {v: k for k, v in attack_categories.items()}

def calculate_false_alarm_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred) 
    num_classes = cm.shape[0] 
    false_alarm_rates = {} 
    for i in range(num_classes):
        # True Negatives (TN): Sum of elements not in row or column of class i 
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        # False Positives (FP): Sum of elements in column of class i excluding the diagonal 
        fp = cm[:, i].sum() - cm[i, i] 
        false_alarm_rate = fp / (fp + tn) 
        false_alarm_rates[attack_categories[i]] = false_alarm_rate 
    return false_alarm_rates

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the data using the imported function
            processed_df = preprocess_data(filepath)

            # Split into features and labels
            y = processed_df['attack_type']
            X = processed_df.drop('attack_type', axis=1)

            # Ensure columns in X match the model's expected features
            expected_features = model.get_booster().feature_names
            for col in expected_features:
                if col not in X.columns:
                    X[col] = 0

            X = X[expected_features]  # Ensure correct column order

            # Make predictions
            predictions = model.predict(X)

            # Classification metrics
            classification_report_str = classification_report(y, predictions, target_names=[attack_categories[i] for i in range(len(attack_categories))])
            confusion_matrix_str = confusion_matrix(y, predictions)
            roc_auc = roc_auc_score(pd.get_dummies(y), model.predict_proba(X), multi_class="ovr")
            mcc = matthews_corrcoef(y, predictions)
            far = calculate_false_alarm_rate(y, predictions)

            # Identify malicious data (non-normal attacks)
            results_df = pd.DataFrame({
                'Packet #': range(1, len(X) + 1),
                'Actual': y.map(attack_categories),
                'Predicted': pd.Series(predictions).map(attack_categories),
                'Is Malicious': ['Yes' if pred != 2 else 'No' for pred in predictions]  # Assuming 'Normal' is class 2
            })

            # Filter only malicious packets
            malicious_df = results_df[results_df['Is Malicious'] == 'Yes']

            # Calculate accuracy
            accuracy = np.mean(y == predictions) * 100

            # Clean up uploaded file
            os.remove(filepath)

            return render_template('results.html',
                                   accuracy=f'{accuracy:.2f}%',
                                   classification_report=classification_report_str,
                                   confusion_matrix=confusion_matrix_str,
                                   roc_auc=f'{roc_auc:.2f}',
                                   mcc=f'{mcc:.2f}',
                                   false_alarm_rates=far,
                                   results=results_df.to_html(classes='table table-striped', index=False),
                                   malicious=malicious_df.to_html(classes='table table-danger', index=False))

        except Exception as e:
            # Clean up uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return f'Error in processing: {str(e)}', 400

if __name__ == '__main__':
    app.run(debug=True)
