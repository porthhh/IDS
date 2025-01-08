import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import joblib
import os

# Define attack categories
attack_categories = {
    "Normal": ["normal", "BENIGN"],
    "Denial of Service (DoS/DDoS and Botnet)": [
        "back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb",
        "processtable", "udpstorm", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest",
        "DoS slowloris", "botnet"
    ],
    "Brute Force": ["ftp-patator", "ssh-patator", "guess_passwd"],
    "Reconnaissance": ["ipsweep", "nmap", "portsweep", "satan", "probe", "analysis", "reconnaissance", "saint", "mscan"],
    "Remote to Local (R2L)": [
        "ftp_write", "imap", "multihop", "phf", "spy", "sendmail", "named", "httptunnel",
        "warezclient", "warezmaster", "infiltration", "snmpgetattack", "snmpguess", "xlock", "xsnoop"
    ],
    "Exploits": ["heartbleed", "shellcode", "backdoor", "exploit"],
    "Fuzzers": ["fuzzers"],
    "User to Root (U2R)": ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"],
    "SQL Injection": ["sqlinjection"],
    "Cross-Site Scripting (XSS)": ["xss"],
    "Generic": ["generic"]
}
attack_to_category = {attack: category for category, attacks in attack_categories.items() for attack in attacks}

# Function to label attacks by category
def label_attack_category(attack):
    return attack_to_category.get(attack, "Unknown")

# Preprocessing function
def preprocess_data(csv_file_path):
    # Load dataset
    df = pd.read_csv(csv_file_path)

    # Define feature groups
    binary_features = ['land', 'logged_in', 'root_shell', 'is_host_login', 'is_guest_login']
    categorical_features = ['protocol_type', 'service', 'flag']
    discrete_features = ['wrong_fragment', 'urgent', 'hot', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                         'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                         'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                         'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                         'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'su_attempted']
    continuous_features = ['duration', 'src_bytes', 'dst_bytes', 'num_failed_logins', 'num_compromised', 'num_root',
                           'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                           'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
    features_to_scale = binary_features + discrete_features + continuous_features

    # Load or initialize scaler
    scaler_path = 'scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        joblib.dump(scaler, scaler_path)

    # Scale features
    df[features_to_scale] = scaler.transform(df[features_to_scale])

    # Optimize service grouping using np.where
    threshold = 1000
    service_counts = df['service'].value_counts()
    df['service_grouped'] = np.where(
        df['service'].map(service_counts) >= threshold, df['service'], 'Other'
    )

    # One-hot encode categorical features in one step
    encoder_path = 'encoder.pkl'
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
    else:
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoder.fit(df[['flag', 'protocol_type', 'service_grouped']])
        joblib.dump(encoder, encoder_path)

    encoded_features = encoder.transform(df[['flag', 'protocol_type', 'service_grouped']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['flag', 'protocol_type', 'service_grouped']))

    # Combine with the original DataFrame
    df = pd.concat([df.drop(['flag', 'protocol_type', 'service', 'service_grouped'], axis=1), encoded_df], axis=1)

    # Map attack categories and filter unknowns
    df['Attack_Category'] = df['class'].apply(label_attack_category)
    df = df[df['Attack_Category'] != "Unknown"]

    # Encode attack categories
    label_encoder_path = 'label_encoder.pkl'
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(df['Attack_Category'])
        joblib.dump(label_encoder, label_encoder_path)

    df['attack_type'] = label_encoder.transform(df['Attack_Category'])
    df.drop(['class', 'Attack_Category'], axis=1, inplace=True)

    # Cleanup intermediate variables
    del service_counts, encoded_features, encoded_df

    return df