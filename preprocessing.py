from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def build_preprocessing_pipeline():
    numeric_features = ['loanAmount', 'apr', 'nPaidOff', 'leadCost']
    categorical_features = ['state', 'payFrequency']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Ensure object types are handled
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def preprocess_data(df):
    initial_row_count = df.shape[0]
    print(f"Initial row count: {initial_row_count}")

    # Create the models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Data Cleaning: Handle missing values
    df.dropna(subset=['loanStatus'], inplace=True)
    after_dropna_count = df.shape[0]
    print(f"Row count after dropping missing 'loanStatus': {after_dropna_count}")

    df['state'] = df['state'].fillna('Unknown')

    # Encode categorical variables
    loan_status_encoder = LabelEncoder()
    df['loanStatus'] = loan_status_encoder.fit_transform(df['loanStatus'])
    
    # Save the loanStatus LabelEncoder
    joblib.dump(loan_status_encoder, 'models/label_encoder.pkl')

    # Separate LabelEncoder for the 'state' column
    state_encoder = LabelEncoder()
    df['state'] = state_encoder.fit_transform(df['state'].astype(str))
    
    # Save the state LabelEncoder as well
    joblib.dump(state_encoder, 'models/state_encoder.pkl')

    # Remove classes with only one instance
    class_counts = df['loanStatus'].value_counts()
    print(f"Class counts before removal: {class_counts}")
    classes_to_keep = class_counts[class_counts > 1].index
    df = df[df['loanStatus'].isin(classes_to_keep)]
    after_class_removal_count = df.shape[0]
    print(f"Row count after removing rare classes: {after_class_removal_count}")

    # Adjust labels to be zero-based
    label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(df['loanStatus'].unique()))}
    df['loanStatus'] = df['loanStatus'].map(label_map)

    # Feature engineering
    df['is_monthly_payment'] = df['payFrequency'].apply(lambda x: 1 if x == 'M' else 0)
    df['loan_to_payment_ratio'] = df['loanAmount'] / df['originallyScheduledPaymentAmount']

    final_row_count = df.shape[0]
    print(f"Final row count after all preprocessing: {final_row_count}")

    return df


