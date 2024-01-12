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
import os

from dataset import TimeSeriesDataset, TimeSeriesAvgDataset
from models import *
from losses import *

input_length = 100
input_cols = ['u', 'v', 'w']
learning_rate = 1e-4
max_epochs = 30
output_len = 10
diff_order = 0
hidden_size = 50
num_layers = 2


train_ds = TimeSeriesDataset('sample_data.csv', input_cols, input_length, y_len = output_len, diff_order=diff_order, train=True, size=10000)
test_ds =  TimeSeriesDataset('sample_data.csv', input_cols, input_length, y_len = output_len, diff_order=diff_order, train=False, size=10000)

train_loader = DataLoader(train_ds, batch_size = 64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size = 64, shuffle=True)

print(len(train_loader))
print(len(test_loader))
df = pd.read_csv('sample_data.csv')
print(len(df))
tpm = np.load('tpm.npy')
# my_model = LSTMModel(len(input_cols), input_length, output_length=output_len, hidden_dim=hidden_size, num_layers=num_layers)
my_model = TransformerModel2(len(input_cols), input_length, output_length=output_len, hidden_size=hidden_size, num_layers=num_layers, lags=False)
# my_model = LinearRegressionFlat(len(input_cols), input_length, output_length=output_len)
optimizer = optim.Adam(my_model.parameters(), lr=1e-4)
time = str(datetime.datetime.now())
path = "experiments/VK_" + my_model.name + "_" + time[:-7]
path = path.replace(" ", "_")
os.system(f"mkdir {path}")
weights_path = path + "/weights.pt"
log_path = path + "/log.csv"
params_path = path + "/params.csv"
params = {"input_length": [input_length], 
          "input_width": [len(input_cols)], 
          "output_length": [output_len], 
          "diff_order": [diff_order], 
          "max_epochs": [max_epochs], 
          "learning_rate": [learning_rate],
          "hidden_size": [hidden_size],
          "num_layers": [num_layers]}
params_df = pd.DataFrame(params)
params_df.to_csv(params_path)

num_epochs = 30
epochs = []
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    my_model.train()
    train_loss = 0
    
    print(len(train_loader))
    cnt = 0
    for batch_X, batch_y in train_loader:
        cnt += 1
        print(cnt)
        optimizer.zero_grad()
        
        outputs = my_model.train_forward(batch_X, batch_y)
        # print(y_pred.shape, batch_y.shape)
        # y_pred = torch.mean(y_pred, 1)
        # batch_y = torch.mean(batch_y, 1)
        # print(y_pred.shape, batch_y.shape)
        # print(y_pred.shape, batch_y.shape)
        
        # print("Y true:", batch_y.shape)
        # print("Y pred:", y_pred.shape)
        # print(y_pred)
        # print(y_pred.shape)
        # print(y_pred, batch_y)
        # print(y_pred.shape)
        # print(y_pred, batch_y)
        loss = outputs.loss
        # print(loss)
        loss.backward()
        optimizer.step()
        # print(loss.item())
        train_loss += loss.item()
        print(loss.item())
    #break
    
    test_loss = 0
    my_model.train(mode=False)
    c2 = 0
    size = 1
    print(len(test_loader))
    for batch_X, batch_y in test_loader:
        y_pred = my_model(batch_X)
        print(c2)
        loss = MSE_loss(y_pred, batch_y)
        print(loss.item())
        test_loss += loss.item()
        c2 += 1
        if c2 > size:
            break
    


    train_loss /= len(train_loader)
    test_loss /= size

    epochs.append(epoch)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}")
    
    if epoch < 2 or test_losses[-1] < test_losses[-2]:
        torch.save(my_model.state_dict(), weights_path)
    
    pd.DataFrame.from_dict({'epochs': epochs, 'train_losses': train_losses, 'test_losses': test_losses}).to_csv(log_path, index=False)
    if epoch > 1 and abs(train_losses[-1] - train_losses[-2]) < 0.0001:
        break

