from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from preprocess import preprocess_data
import joblib
from dashboard import create_dashboard, global_store
import json

app = Flask(__name__)

# Initialize the Dash app
dash_app = create_dashboard(app)

# No need to register the blueprint - Dash handles this internally

# Define attack categories mapping
attack_categories = {
    0: 'Brute Force',
    1: 'Denial of Service (DoS/DDoS and Botnet)',
    2: 'Normal',
    3: 'Reconnaissance',
    4: 'Remote to Local (R2L)',
    5: 'User to Root (U2R)'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    # Save the uploaded file temporarily
    temp_path = 'temp_upload.csv'
    file.save(temp_path)

    # Preprocess the data
    df = preprocess_data(temp_path)
    
    # Load the model
    model = joblib.load('nslkdd_xgb.pkl')
    
    # Get features and true labels
    X = df.drop('attack_type', axis=1)
    y_true = df['attack_type']
    
    # Ensure columns in X match the model's expected features
    expected_features = model.get_booster().feature_names
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_features]
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Convert numerical predictions to category names
    categories = [attack_categories[i] for i in range(len(attack_categories))]
    
    # Prepare metrics for dashboard
    class_metrics = []
    for i, category in enumerate(categories):
        if str(i) in report:
            metrics = report[str(i)]
            metrics['category'] = category
            metrics['accuracy'] = report['accuracy']
            class_metrics.append(metrics)

    dashboard_data = {
        'class_metrics': class_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'macro_avg': report['macro avg'],
        'weighted_avg': report['weighted avg']
    }

    # Store the metrics in the global variable
    global global_store
    global_store = dashboard_data

    # Create results table
    results_df = pd.DataFrame({
        'True Label': [attack_categories[int(x)] for x in y_true],
        'Predicted Label': [attack_categories[int(x)] for x in y_pred]
    })
    
    # Filter malicious traffic (non-normal)
    malicious_df = results_df[results_df['Predicted Label'] != 'Normal']
    
    return render_template('results.html',
                         accuracy=f"{report['accuracy']:.2%}",
                         results=results_df.to_html(classes='table table-striped', index=False),
                         malicious=malicious_df.to_html(classes='table table-striped', index=False))

if __name__ == '__main__':
    app.run(debug=True)
