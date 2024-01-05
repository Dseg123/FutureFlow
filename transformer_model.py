import torch
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from dataset import TimeSeriesDataset
from models import LinearRegression

x_len = 50
y_len = 5

train_ds = TimeSeriesDataset('sample_data.csv', ['u', 'v', 'w'], x_len=x_len, y_len=y_len, train=True)
test_ds =  TimeSeriesDataset('sample_data.csv', ['u', 'v', 'w'], x_len=x_len, y_len=y_len, train=False)

train_loader = DataLoader(train_ds, batch_size = 64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size = 64, shuffle=True)



configuration = TimeSeriesTransformerConfig(context_length = x_len//2, prediction_length=y_len, input_size=3, num_time_features=1)
model = TimeSeriesTransformerForPrediction(configuration)


for epoch in range(30):
    avg_loss = 0
    num_batch = 0
    print(len(train_loader))
    for batch_x, batch_y in train_loader:
        print(num_batch)
        
        past_time_features = torch.zeros((batch_x.shape[0], batch_x.shape[1], 1))
        for i in range(past_time_features.shape[1]):
            past_time_features[:, i, 0] = i
        # print(past_time_features)

        past_observed_mask = torch.ones_like(batch_x)

        future_time_features = torch.zeros((batch_y.shape[0], batch_y.shape[1], 1))
        for i in range(future_time_features.shape[1]):
            future_time_features[:, i, 0] = x_len + i
        # print(future_time_features)

        outputs = model(past_values=batch_x, past_time_features=past_time_features, past_observed_mask=past_observed_mask, future_values=batch_y, future_time_features=future_time_features)
        loss = outputs.loss
        print(loss)
        loss.backward()

        avg_loss += loss.detach().item()
        num_batch += 1
    print("Epoch", epoch)
    print("Loss", avg_loss/num_batch)

    
    