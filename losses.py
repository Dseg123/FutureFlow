import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


def MSE_loss(y_pred, y_true):
    # print(y_pred.shape, y_true.shape)
    y_pred = torch.reshape(y_pred, y_true.shape)
    # print("PRED", y_pred)
    # print("TRUE", y_true)
    return torch.mean(torch.square(y_pred - y_true))

def MAE_loss(y_pred, y_true):
    y_pred = torch.reshape(y_pred, y_true.shape)

    return torch.mean(torch.abs(y_pred - y_true))

def ratio_loss(y_pred, y_true):
    y_pred = torch.reshape(y_pred, y_true.shape)

    out = torch.div(y_pred - y_true, y_true + 10**(-7))
    return torch.mean(torch.square(out))
