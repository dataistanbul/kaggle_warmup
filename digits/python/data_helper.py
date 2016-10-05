import pandas as pd
import numpy as np
from math import ceil

 
def load_train_data(data_path):
    p=0.8 # percentage for training
    df = pd.read_csv(data_path)
    df = df.reindex(np.random.permutation(df.index)) # Other option
    #df = df.sample(frac=1, axis=0)
    marker = int(ceil(df.shape[0] * p))
    X_train = df.iloc[0:marker,1:].values
    y_train = df.iloc[0:marker,0].values
    X_test = df.iloc[marker:,1:].values
    y_test = df.iloc[marker:,0].values
    return (X_train, y_train), (X_test, y_test)
    
def load_test_data(data_path):
    df = pd.read_csv(data_path)
    X_test = df.values
    return X_test

def save_preds(predictions, data_path):
    submission = pd.DataFrame({ 'ImageId': range(1,  len(predictions)+1), 'Label': predictions })
    submission.to_csv(data_path, index=False)
