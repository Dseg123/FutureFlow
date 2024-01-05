import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

def display_progress_old(experiment_log):
    log = pd.read_csv(experiment_log)
    train_losses = log['train_losses']
    test_losses = log['test_losses']
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.title("Model Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()

def display_progress(experiment_path):
    log = pd.read_csv(experiment_path + '/log.csv')
    train_losses = log['train_losses']
    test_losses = log['test_losses']
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.title("Model Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()


def get_preds(model, data_df, input_length, input_cols, thresh = 2000):
    arr = data_df[input_cols].to_numpy()
    preds = []
    trues = []
    print(len(arr))
    for i in range(max(0, len(arr) - thresh - input_length), len(arr) - input_length):
        print(i)
        X = arr[i:i+input_length, :]
        y = arr[i+input_length, :]

        X_ten = torch.tensor(X)
        X_ten = X_ten.float()
        X_ten = X_ten.reshape((1, X_ten.shape[0], X_ten.shape[1]))

        y_ten = torch.tensor(y).float()
        
        pred = (model(X_ten)).detach().numpy()
        pred = pred.tolist()
        true = y_ten.numpy()
        true = true.tolist()
        preds.append(pred)
        trues.append(true)
        # preds = np.concatenate((preds, pred), axis=0)
    return preds, trues

def get_preds(model, data_df, input_length, input_cols, thresh = 2000):
    arr = data_df[input_cols].to_numpy()
    preds = []
    trues = []
    print(len(arr))
    for i in range(max(0, len(arr) - thresh - input_length), len(arr) - input_length):
        print(i)
        X = arr[i:i+input_length, :]
        

        X_ten = torch.tensor(X)
        X_ten = X_ten.float()
        X_ten = X_ten.reshape((1, X_ten.shape[0], X_ten.shape[1]))

        
        
        pred = (model(X_ten)).detach().numpy()
        # print(pred.shape)
        pred_len = pred.shape[1]
        pred = np.mean(pred, axis=1)
        # print(pred.shape)

        pred = pred.tolist()

        y = arr[i+input_length:min(i+input_length+pred_len, len(arr)), :]
        y_ten = torch.tensor(y).float()
        true = y_ten.numpy()
        # print(true)
        # print(true.shape)
        true = np.mean(true, axis=0)
        # print(true.shape)
        # print(true)
        true = true.tolist()
        
        preds.append(pred)
        trues.append(true)
        # preds = np.concatenate((preds, pred), axis=0)
    return preds, trues

def display_preds(model, data_df, input_length, input_cols, thresh = 2000):
    preds, trues = get_preds(model, data_df, input_length, input_cols, thresh)
    preds = np.array(preds)
    trues = np.array(trues)
    print(preds.shape)
    print(trues.shape)
    plt.plot(preds[:100, 0, 0], label='Predicted')
    plt.plot(trues[:100, 0], label='Actual')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Wind Speed")
    plt.title(model.name + " Time Series Predictions")
    plt.show()