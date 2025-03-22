from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
import boto3
import pandas as pd
from io import StringIO, BytesIO
import mlflow
import xgboost as xgb
import logging
import os
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib 
import mlflow.pyfunc 
from sklearn.metrics import mean_squared_error
from math import sqrt
from imblearn.over_sampling import SMOTE
import numpy as np


BUCKET_NAME = 's3ml'
INPUT_FILE_KEY = 'IMDB-Dataset.csv'
PROCESSED_FILE_KEY = 'processed/output.parquet'
#MLFLOW_TRACKING_URI = 'http://mlflow_server:5000'
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000"))
MLFLOW_EXPERIMENT = 'ml_pipeline_experiment'
AWS_CONN_ID = 'mlconn'
AWS_REGION = 'us-east-1'


def get_s3_client():
    conn = BaseHook.get_connection(AWS_CONN_ID)
    
    # Set AWS credentials for boto3 and MLflow
    os.environ['AWS_ACCESS_KEY_ID'] = conn.login
    os.environ['AWS_SECRET_ACCESS_KEY'] = conn.password
    os.environ['AWS_DEFAULT_REGION'] = AWS_REGION
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://s3.amazonaws.com'  # Remove or change if not using AWS S3

    return boto3.client(
        's3',
        aws_access_key_id=conn.login,
        aws_secret_access_key=conn.password,
        region_name=AWS_REGION
    )


def extract_data():
    logging.info("Starting data extraction")
    s3 = get_s3_client()
    response = s3.get_object(Bucket=BUCKET_NAME, Key=INPUT_FILE_KEY)
    df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    logging.info(f"Extracted data with shape: {df.shape}")
    
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=PROCESSED_FILE_KEY, Body=buffer.getvalue())
    logging.info("Saved extracted data as Parquet to S3")


def preprocess_data():
    logging.info("Starting data preprocessing")
    
    s3 = get_s3_client()
    response = s3.get_object(Bucket=BUCKET_NAME, Key=PROCESSED_FILE_KEY)
    df = pd.read_parquet(BytesIO(response['Body'].read()))
    logging.info(f"Loaded data with shape: {df.shape}")

    # Check and rename sentiment column
    if 'sentiment' not in df.columns:
        raise ValueError("The 'sentiment' column is missing in the dataset.")
    df = df.rename(columns={'sentiment': 'target'})

    # Encode target labels (positive/negative -> 1/0)
    df['target'] = LabelEncoder().fit_transform(df['target'])

    # Assuming the first column is the review text
    text_column = df.columns[0]
    df[text_column] = df[text_column].astype(str)

    # TF-IDF vectorization of the reviews
    vectorizer = TfidfVectorizer(max_features=500)
    X_tfidf = vectorizer.fit_transform(df[text_column]).toarray()
    # Save TF-IDF vectorizer to S3
    vectorizer_buffer = BytesIO()
    joblib.dump(vectorizer, vectorizer_buffer)
    vectorizer_buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key='processed/tfidf_vectorizer.joblib', Body=vectorizer_buffer.getvalue())
    logging.info("Saved TF-IDF vectorizer to S3")
    
    # Combine TF-IDF features with target
    processed_df = pd.DataFrame(X_tfidf, columns=[f'tfidf_{i}' for i in range(X_tfidf.shape[1])])
    processed_df['target'] = df['target'].values
    
    # Log class distribution before SMOTE
    class_counts = processed_df['target'].value_counts()
    logging.info(f"Class distribution before SMOTE: {class_counts.to_dict()}")

    # Save preprocessed data back to S3
    buffer = BytesIO()
    processed_df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=PROCESSED_FILE_KEY, Body=buffer.getvalue())

    logging.info("Saved preprocessed data as Parquet to S3")


def apply_smote_and_train_model():
    logging.info("Starting model training with SMOTE")

    # Load preprocessed data from S3
    s3 = get_s3_client()
    response = s3.get_object(Bucket=BUCKET_NAME, Key=PROCESSED_FILE_KEY)
    df = pd.read_parquet(BytesIO(response['Body'].read()))
    logging.info(f"Loaded preprocessed data with shape: {df.shape}")

    if 'target' not in df.columns:
        raise ValueError("The 'target' column is missing in the dataset.")

    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Apply SMOTE to balance the classes
    logging.info("Applying SMOTE to balance class distribution")
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    # Log resampled class distribution
    unique, counts = np.unique(y_smote, return_counts=True)
    class_counts_smote = dict(zip(unique, counts))
    logging.info(f"Class distribution after SMOTE: {class_counts_smote}")
    logging.info(f"Data shape after SMOTE: X={X_smote.shape}, y={y_smote.shape}")

    # Load the same vectorizer from S3
    vectorizer_response = s3.get_object(Bucket=BUCKET_NAME, Key='processed/tfidf_vectorizer.joblib')
    vectorizer = joblib.load(BytesIO(vectorizer_response['Body'].read()))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    class SentimentModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import joblib
            self.vectorizer = joblib.load(context.artifacts["vectorizer"])
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(context.artifacts["xgb_model"])

        def predict(self, context, model_input):
            X_transformed = self.vectorizer.transform(model_input['review_text']).toarray()
            dmatrix = xgb.DMatrix(X_transformed)
            preds = self.model.predict(dmatrix)
            return (preds > 0.5).astype(int)

    with mlflow.start_run() as run:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'error', 'rmse'],  # Added RMSE as evaluation metric
            'learning_rate': 0.1,
            'max_depth': 6
        }

        # Split data for evaluation (80% train, 20% validation)
        train_size = int(0.8 * len(X_smote))
        X_train, X_val = X_smote[:train_size], X_smote[train_size:]
        y_train, y_val = y_smote[:train_size], y_smote[train_size:]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evals_result = {}

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=50,
            evals=[(dtrain, 'train'), (dval, 'val')],
            evals_result=evals_result,
            verbose_eval=False
        )

        # Calculate and log RMS error
        y_pred = model.predict(dval)
        rms_error = sqrt(mean_squared_error(y_val, y_pred))
        logging.info(f"Root Mean Square Error on validation set: {rms_error}")

        mlflow.log_params(params)
        mlflow.log_param("num_boost_round", 80)
        mlflow.log_param("smote_applied", "True")
        mlflow.log_param("smote_class_distribution", str(class_counts_smote))
        mlflow.log_metric("rms_error", rms_error)

        # Log training metrics
        for step in range(50):
            # Log training metrics
            if 'train' in evals_result and 'logloss' in evals_result['train']:
                logloss = evals_result['train']['logloss'][step]
                mlflow.log_metric("train_logloss", logloss, step=step)
            
            if 'train' in evals_result and 'error' in evals_result['train']:
                error = evals_result['train']['error'][step]
                accuracy = 1 - error
                mlflow.log_metric("train_error", error, step=step)
                mlflow.log_metric("train_accuracy", accuracy, step=step)
            
            if 'train' in evals_result and 'rmse' in evals_result['train']:
                rmse = evals_result['train']['rmse'][step]
                mlflow.log_metric("train_rmse", rmse, step=step)
            
            # Log validation metrics if available
            if 'val' in evals_result and 'logloss' in evals_result['val']:
                val_logloss = evals_result['val']['logloss'][step]
                mlflow.log_metric("val_logloss", val_logloss, step=step)
            
            if 'val' in evals_result and 'error' in evals_result['val']:
                val_error = evals_result['val']['error'][step]
                val_accuracy = 1 - val_error
                mlflow.log_metric("val_error", val_error, step=step)
                mlflow.log_metric("val_accuracy", val_accuracy, step=step)
            
            if 'val' in evals_result and 'rmse' in evals_result['val']:
                val_rmse = evals_result['val']['rmse'][step]
                mlflow.log_metric("val_rmse", val_rmse, step=step)

        # Save vectorizer and model locally
        vectorizer_path = "/tmp/tfidf_vectorizer.joblib"
        xgb_model_path = "/tmp/xgb_model.json"
        joblib.dump(vectorizer, vectorizer_path)
        model.save_model(xgb_model_path)

        # Log the combined PyFunc model
        mlflow.pyfunc.log_model(
            artifact_path="sentiment_pipeline",
            python_model=SentimentModel(),
            artifacts={
                "vectorizer": vectorizer_path,
                "xgb_model": xgb_model_path
            },
            registered_model_name="xgboost_IMDB_model_smote"
        )

        logging.info(f"Combined model registered as 'xgboost_IMDB_model_smote' in MLflow Model Registry")

    logging.info("Model training complete and logged to MLflow")
    
    # Test the model immediately after training
    text = "movie is waste of money"
    X_transformed = vectorizer.transform([text]).toarray()
    dmatrix = xgb.DMatrix(X_transformed)
    prediction = model.predict(dmatrix)
    sentiment = "positive" if prediction[0] > 0.5 else "negative"
    logging.info(f"Test prediction for '{text}': {sentiment} ({prediction[0]})")
    
    # Save RMS error to S3
    rms_data = {'rms_error': rms_error}
    rms_df = pd.DataFrame([rms_data])
    buffer = BytesIO()
    rms_df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key='metrics/rms_error.csv', Body=buffer.getvalue())
    logging.info(f"Saved RMS error ({rms_error}) to S3")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 5),
    'retries': 1,
}

with DAG(
    dag_id='ml_pipeline_dag2',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    extract_task = PythonOperator(
        task_id='extract_csv_from_s3',
        python_callable=extract_data
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    train_task = PythonOperator(
        task_id='train_xgboost_model_with_smote',
        python_callable=apply_smote_and_train_model
    )

    extract_task >> preprocess_task >> train_task