import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

RANDOM_STATE = 42

def load_data(train_path, val_path):

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df.drop(columns=['Class'])
    y_train = train_df['Class']

    X_val = val_df.drop(columns=['Class'])
    y_val = val_df['Class']

    X_train, X_val = scale_features(X_train, X_val)

    return X_train, y_train, X_val, y_val


def scale_features(X_train, X_val):
    
    X_train = X_train.copy()
    X_val = X_val.copy()

    # # Step1: Log Transformation to Amount
    X_train['Amount'] = np.log1p(X_train['Amount'])
    X_val['Amount'] = np.log1p(X_val['Amount'])


    # Step 2: Scale all features using RobustScaler

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val



def get_sampler(technique, sampling_strategy = None, random_state = 42):

    if technique == 'smote':
        return SMOTE(sampling_strategy= sampling_strategy, random_state= random_state)

    if technique == 'random_over':
        return RandomOverSampler(sampling_strategy= sampling_strategy, random_state= random_state)

    if technique == 'random_under':
        return RandomUnderSampler(sampling_strategy= sampling_strategy, random_state= random_state)

    if technique in(None, 'none'):
        return 'passthrough'  # baseline without resampling

    raise ValueError(f'Unknown technique: {technique}')