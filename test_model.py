import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import datetime

from dataset import TimeSeriesDataset
from models import *
from losses import *

input_length = 200
input_cols = ['u', 'v', 'w']

train_ds = TimeSeriesDataset('sample_data.csv', input_cols, input_length, train=True)
test_ds =  TimeSeriesDataset('sample_data.csv', input_cols, input_length, train=False)

train_loader = DataLoader(train_ds, batch_size = 64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size = 64, shuffle=True)

weights = torch.load("/home/dylaneg/Documents/Programming/IROM/FutureFlow/Lin_Reg_Flat_weights_2023-10-05 14:31:44.402433.pt")
trained_model = LinearRegressionFlat(len(input_cols), input_length)
trained_model.load_state_dict(weights)

    
test_loss = 0
trained_model.train(mode=False)
for batch_X, batch_y in test_loader:
    y_pred = trained_model(batch_X)
    loss = MSE_loss(y_pred, batch_y)
    test_loss += loss.item()
    

test_loss /= len(test_loader)
print("Test Loss:", test_loss)


