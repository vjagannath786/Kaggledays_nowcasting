import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np

import config
from dataset import NocDataset
import engine
from model import NoCModel

from sklearn.model_selection import KFold


def preprocess():

    train = pd.read_csv(os.path.join(config.PATH, 'train.csv'))
    sensor = pd.read_csv(os.path.join(config.PATH,'sensor.csv'))
    submission = pd.read_csv(os.path.join(config.PATH,'sample_submission.csv'))

    #tmp = sensor.groupby('chunk_id').size().reset_index().reset_index()
    #del tmp[0]

    #sensor = sensor.merge(tmp, on='chunk_id',how='left')

    sensor.drop(['date','time'],axis=1, inplace=True)

    #tmp = sensor.set_index(['chunk_id','time'])['rain'].unstack().reset_index()
    #tmp.fillna(0, inplace=True)
    train_df = train.drop(['rain'],axis=1).merge(sensor, on='chunk_id', how='left')
    sample_df = submission.drop(['rain_prediction'],axis=1).merge(sensor, on='chunk_id', how='left')

    '''
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

    '''

    return train_df, sample_df





def run_training(train, test):


    x = train.drop(['rain','chunk_id','stationid'], axis = 1)
    y = train['rain']
    x_test = test.drop(['rain','chunk_id','stationid'],axis=1).values

    oof_predictions = np.zeros(x.shape[0])
    # Create test array to store predictions
    test_predictions = np.zeros(x_test.shape[0])
    # Create a KFold object
    kfold = KFold(n_splits = 5, random_state=42, shuffle=True)

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = x.loc[trn_ind], x.loc[val_ind]
        y_train, y_val = y.loc[trn_ind], y.loc[val_ind]


        train_ds = NocDataset(data= x_train, targets=y_train, is_test=False)

        valid_ds = NocDataset(data=x_val, targets=y_val, is_test=False)

        test_ds = NocDataset(data=x_test, targets='',is_test=True)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2,
                          pin_memory=False, drop_last=True)

        val_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2,
                          pin_memory=False, drop_last=False)

        test_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2,
                          pin_memory=False, drop_last=False)

        model = NoCModel()
        model.to(config.DEVICE)

        optimizer = optim.Adam(model.parameters(),lr=config.learning_rate)

        _loss = 1000
        v_preds = []
        t_preds = []

        for epoch in range(config.EPOCHS):
            train_loss = engine.train_fn(model, train_loader, optimizer)
            valid_preds, valid_loss = engine.eval_fn(model, val_loader)
            test_preds,_ = engine.eval_fn(model, test_loader)

            print(f'train loss is {train_loss} and valid loss is {valid_loss}')
            
            if valid_loss < _loss:
                v_preds = valid_preds
                t_preds = test_preds
                _loss = valid_loss
            


        oof_predictions[val_ind] = v_preds

        test_predictions += t_preds


        return oof_predictions, np.array(test_predictions).mean(axis=0)






if __name__ == "__main__":

    
    train, test = preprocess()



    test_predictions = run_training(train, test)

    print(train.loc[train.chunk_id == '2397129cde'])

    print(test.head())