import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import os
import numpy as np

import config
import dataset
import engine
from model import NoCModel

from sklearn.model_selection import KFold


def preprocess():

    train = pd.read_csv(os.path.join(config.PATH, 'train.csv'))
    sensor = pd.read_csv(os.path.join(config.PATH,'sensor.csv'))
    submission = pd.read_csv(os.path.join(config.PATH,'sample_submission.csv'))


    tmp = sensor.set_index(['chunk_id','time'])['rain'].unstack().reset_index()
    tmp.fillna(0, inplace=True)
    train_df = train.merge(tmp, on='chunk_id', how='left')
    sample_df = submission.merge(tmp, on='chunk_id', how='left')

    _aggs = {
          'tempc': [ np.mean],
          'feelslikec':[np.min, np.max, np.mean],
         'windspeed_kph':[np.mean],
         'windgust_kph':[np.min,np.max,np.mean],
         'pressure_mb':[np.min, np.max],
        'humidity': [np.min, np.max, np.mean],
        'visibility_km': [np.min, np.max, np.mean],
		'cloud':[np.mean],
		'heatindexc':[np.mean],
		'dewpointc':[np.mean],
		'windchillc':[np.mean],
		'uvindex':[np.mean],
		
        
        }
    
    tmp_sensor = sensor.groupby(['chunk_id']).agg(_aggs).reset_index()

    tmp_sensor = sensor.groupby(['chunk_id']).agg(_aggs).reset_index()

    train_df = train_df.merge(tmp_sensor, left_on='chunk_id', right_on='chunk_id', how='left')

    sample_df = sample_df.merge(tmp_sensor, left_on='chunk_id', right_on='chunk_id', how='left')

    return train_df, sample_df





def run_training(train, test):


    x = train.drop(['rain','chunk_id'], axis = 1)
    y = train['rain']
    x_test = test.drop(['rain_prediction','chunk_id'],axis=1)

    oof_predictions = np.zeros(x.shape[0])
    # Create test array to store predictions
    test_predictions = np.zeros(x_test.shape[0])
    # Create a KFold object
    kfold = KFold(n_splits = 5, random_state=42, shuffle=True)

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]


        train_ds = NocDataset(data= x_train, targets=y_train)

        valid_ds = NocDataset(data=x_val, targets=y_val)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=NW,
                          pin_memory=False, drop_last=True)

        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=NW,
                          pin_memory=False, drop_last=False)




if __name__ == "__main__":

    
    train, test = preprocess()



    test_predictions = run_training(train, test)