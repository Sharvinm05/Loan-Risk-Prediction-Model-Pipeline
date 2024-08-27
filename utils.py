import logging

def setup_logging():
    logging.basicConfig(filename='pipeline_logs.log', level=logging.INFO)
    logging.info('Logging setup completed.')

def check_data(df):
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")
    logging.info('Data check passed.')
