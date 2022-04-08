import torch

import pandas as pd

from dataset import NocDataset


df = pd.read_csv('../../sensor.csv')


dataset = NocDataset(data=df.drop(['chunk_id','stationid','date','time','rain'], axis=1).values, targets= df['rain'].values)


#dataloader = DataLoader(dataset,5)
print(df.drop(['chunk_id','stationid','date','time','rain'], axis=1).columns)