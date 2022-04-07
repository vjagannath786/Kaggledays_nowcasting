import torch
from torch.utils.data import Dataloader
import pandas as pd

from dataset import NocDataset


df = pd.read_csv('../../input/nowcastingweather/sensor.csv')


dataset = NocDataset(data=df.drop(['chunk_id','stationid','date','time','rain'], axis=1).values, targets= df['rain'].values)


dataloader = Dataloader(dataset, )
print(dataset[0])