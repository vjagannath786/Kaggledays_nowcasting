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
from sklearn.metrics import confusion_matrix


def quadKappa(act,pred,n=4,hist_range=(0,3)):
    
    O = confusion_matrix(act,pred)
    O = np.divide(O,np.sum(O))
    
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            W[i][j] = ((i-j)**2)/((n-1)**2)
            
    act_hist = np.histogram(act,bins=n,range=hist_range)[0]
    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]
    
    E = np.outer(act_hist,prd_hist)
    E = np.divide(E,np.sum(E))
    
    num = np.sum(np.multiply(W,O))
    den = np.sum(np.multiply(W,E))
        
    return 1-np.divide(num,den)


def preprocess():

    train = pd.read_csv(os.path.join(config.PATH, 'train.csv'))
    sensor = pd.read_csv(os.path.join(config.PATH,'sensor.csv'))
    submission = pd.read_csv(os.path.join(config.PATH,'sample_submission.csv'))

    #tmp = sensor.groupby('chunk_id').size().reset_index().reset_index()
    #del tmp[0]

    #sensor = sensor.merge(tmp, on='chunk_id',how='left')

    sensor["days"] = (pd.to_datetime(sensor["date"]) - pd.to_datetime("2021-01-01")).dt.days
    sensor = sensor.sort_values(["stationid", "days", "time"]).reset_index(drop=True)

    numeric_features = [col for col in sensor.columns if sensor[col].dtype == np.float64]
    numeric_features += ["uvindex", "isday", "time", "days", "rain"]

    #sensor.drop(['date','time'],axis=1, inplace=True)

    #tmp = sensor.set_index(['chunk_id','time'])['rain'].unstack().reset_index()
    #tmp.fillna(0, inplace=True)
    #train_df = train.drop(['rain'],axis=1).merge(sensor, on='chunk_id', how='left')
    #sample_df = submission.drop(['rain_prediction'],axis=1).merge(sensor, on='chunk_id', how='left')


    F_matrix = sensor[numeric_features].values.reshape(sensor.shape[0]//23, 23, len(numeric_features))
    

    index_df = sensor[["chunk_id", "stationid"]].drop_duplicates()
    index_df["idx"] = np.arange(index_df.shape[0])
    

    train = train.merge(index_df, on="chunk_id")

    submission = submission.merge(index_df, on="chunk_id")

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

    return F_matrix, train, submission





def run_training(train, test):


    x = train['idx']
    y = train['rain']
    x_test = test['idx'].values

    oof_predictions = np.zeros(x.shape[0])
    # Create test array to store predictions
    test_predictions = np.zeros(x_test.shape[0])
    # Create a KFold object
    kfold = KFold(n_splits = 5, random_state=42, shuffle=True)

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = x.iloc[trn_ind].values, x.iloc[val_ind].values
        y_train, y_val = y.iloc[trn_ind].values, y.iloc[val_ind].values


        train_ds = NocDataset(F_matrix=F_matrix,data= x_train, targets=y_train, is_test=False)

        valid_ds = NocDataset(F_matrix=F_matrix,data=x_val, targets=y_val, is_test=False)

        test_ds = NocDataset(F_matrix,x_test, '',is_test=True)

        #print(test_ds)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2,
                          pin_memory=False, drop_last=True)

        val_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2,
                          pin_memory=False, drop_last=False)

        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2,
                          pin_memory=False, drop_last=False)

        model = NoCModel()
        model.to(config.DEVICE)

        optimizer = optim.Adam(model.parameters(),lr=config.learning_rate, betas=(0.9, 0.999),
                                 eps=1e-08)

        _loss = 1000
        v_preds = []
        t_preds = []

        for epoch in range(config.EPOCHS):
            train_loss = engine.train_fn(model, train_loader, optimizer)
            valid_preds, valid_loss = engine.eval_fn(model, val_loader)
            test_preds = engine.pred_fn(model, test_loader)

            print(f'train loss is {train_loss} and valid loss is {valid_loss}')
            
            if valid_loss < _loss:
                v_preds = valid_preds
                t_preds = test_preds
                _loss = valid_loss
            

        #print(np.concatenate(v_preds,axis=0))
        a = np.concatenate(v_preds, axis=0)
        b = np.concatenate(a, axis=0)
        oof_predictions[val_ind] = b

        c = np.concatenate(t_preds, axis=0)
        d = np.concatenate(c, axis=0)


        print(d)
        test_predictions += np.clip(np.round(d),0,3)


    return oof_predictions, [x/5 for x in test_predictions]






if __name__ == "__main__":

    
    F_matrix, train, test = preprocess()

    print(test.shape)

    oof_predictions, test_predictions = run_training(train, test)

    #print(train.loc[train.chunk_id == '2397129cde'])

    print(quadKappa(train["rain"].values, np.clip(np.round(oof_predictions), 0, 3)))
    
    test['rain_prediction'] = np.round(test_predictions).astype('int')

    test.to_csv('../../working/submission_v1.csv', index=False, columns=['chunk_id', 'rain_prediction'])

    print(test['rain_prediction'].unique())