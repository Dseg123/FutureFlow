import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig


class NaiveModel(nn.Module):
    def __init__(self, input_width, input_length, output_length = 1):
        super(NaiveModel, self).__init__()
        self.name = "Naive"
        self.width = input_width
        self.length = input_length
        self.output_length = output_length

    def forward(self, x):
        last = x[:, -1, :]
        outs = torch.zeros((x.shape[0], self.output_length, self.width))
        for i in range(x.shape[0]):
            for j in range(self.output_length):
                for k in range(self.width):
                    outs[i, j, k] = last[i, k]
        return outs

# Define the logistic regression model
class LinearRegression(nn.Module):

    def __init__(self, input_width, input_length, output_length = 1):
        super(LinearRegression, self).__init__()
        self.name = "Lin_Reg"
        self.width = input_width
        self.length = input_length
        self.output_length = output_length
        self.linears = nn.ModuleList([nn.Linear(input_length, output_length) for i in range(input_width)])


    def forward(self, x):
        a = self.linears[0](x[:, :, 0])
        # print(torch.unsqueeze(a, 2).shape)
        pred = torch.cat([torch.unsqueeze(self.linears[i](x[:, :, i]), 2) for i in range(self.width)], 2)
        # print(pred.shape)
        return pred
    
    
class LinearRegressionFlat(nn.Module):

    def __init__(self, input_width, input_length, output_length = 1):
        super(LinearRegressionFlat, self).__init__()
        self.name = "Lin_Reg_Flat"
        self.width = input_width
        self.length = input_length
        self.output_length = output_length
        self.linear = nn.Linear(input_width * input_length, input_width * output_length)


    def forward(self, x):
        y = self.linear(torch.reshape(x, (-1, self.width * self.length)))
        return torch.reshape(y, (-1, self.output_length, self.width))
    
    def custom_loss(self, y_pred, y_true):
        return torch.mean(torch.square(y_pred - y_true))
    
class DeepRegression(nn.Module):
    def __init__(self, input_width, input_length, output_length=1, layer_sizes = [100, 20]):
        super(DeepRegression, self).__init__()
        self.name = "Deep_Reg"
        self.width = input_width
        self.length = input_length
        self.output_length = output_length
        layer_list = []
        curr_size = input_width * input_length
        for layer in layer_sizes:
            layer_list.append(nn.Linear(curr_size, layer))
            curr_size = layer
        layer_list.append(nn.Linear(curr_size, input_width * output_length))
        self.linears = nn.ModuleList(layer_list)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = torch.reshape(x, (-1, self.width * self.length))
        for i in range(len(self.linears)):
            layer = self.linears[i]
            x = layer(x)
            if i != len(self.linears) - 1:
                x = self.activation(x)
        
        return torch.reshape(x, (-1, self.output_length, self.width))

class RecurrentModel(nn.Module):
    def __init__(self, input_width, input_length, output_length = 1, hidden_size = None, num_layers = 1):
        super(RecurrentModel, self).__init__()
        self.name = "RNN"

        self.input_width = input_width
        self.input_length = input_length
        self.num_layers = num_layers
        self.output_length = output_length

        self.hidden_dim = hidden_size
        if self.hidden_dim == None:
            self.hidden_dim = input_width


        self.rnn = nn.RNN(input_size=self.input_width, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.input_width)

    def forward_once(self, x):
        x, _ = self.rnn(x)
        x_last = x[:, -1, :]
        x_last = self.linear(x_last)
        return x_last

    def forward(self, x):
        y = torch.clone(x)
        final = torch.zeros((x.shape[0], self.output_length, x.shape[2]))
        for i in range(self.output_length):
            out = self.forward_once(y)
            final[:, i, :] =  out
            y[:, :-1, :] = y[:, 1:, :]
            y[:, -1, :] = out

        return final
    
class LSTMModel(nn.Module):
    def __init__(self, input_width, input_length, output_length = 1, hidden_size = None, num_layers = 1):
        super(LSTMModel, self).__init__()
        self.name = "LSTM"

        self.input_width = input_width
        self.input_length = input_length
        self.num_layers = num_layers
        self.output_length = output_length

        self.hidden_dim = hidden_size
        if self.hidden_dim == None:
            self.hidden_dim = input_width


        self.lstm = nn.LSTM(input_size=self.input_width, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.input_width)

    def forward_once(self, x):
        z, _ = self.lstm(x)
        z_last = z[:, -1, :]
        lin_out = self.linear(z_last)
        return lin_out

    def forward(self, x):
        y = torch.clone(x)
        final = torch.zeros((x.shape[0], self.output_length, x.shape[2]))
        for i in range(self.output_length):
            out = self.forward_once(y)
            final[:, i, :] =  out
            y[:, :-1, :] = y[:, 1:, :]
            y[:, -1, :] = out

            # print(y.shape)
            # print(out.shape)

        return final


class TransformerModel(nn.Module):

    def __init__(self, input_width, input_length, lags=False, context_length=50, num_lags = 10, output_length = 1, hidden_size=64, num_layers=2):
        super(TransformerModel, self).__init__()

        self.name = "Transformer"
        self.width = input_width
        self.length = input_length
        self.output_length = output_length
        
        if lags:
            self.lags_sequence = [i * (input_length - context_length)//num_lags for i in range(num_lags)]
            self.context_length = context_length
        else:
            self.lags_sequence = [1, 2, 3, 4, 5, 6, 7]
            self.context_length = input_length//2


        configuration = TimeSeriesTransformerConfig(prediction_length=output_length, 
                                                    input_size=input_width, 
                                                    context_length=self.context_length, 
                                                    num_time_features=1, 
                                                    lags_sequence = self.lags_sequence,
                                                    d_model=hidden_size, 
                                                    decoder_layers=num_layers, 
                                                    encoder_layers=num_layers, 
                                                    output_hidden_states=True)

        self.transformer = TimeSeriesTransformerForPrediction(configuration)
        self.linear = nn.Linear(hidden_size, input_width)

        self.skip = nn.Linear(input_width * self.context_length, input_width * output_length)
        self.last = nn.Linear(2*input_width, input_width)

    def forward(self, x):
        past_time_features = torch.zeros((x.shape[0], x.shape[1], 1))
        for i in range(past_time_features.shape[1]):
            past_time_features[:, i, 0] = i

        past_observed_mask = torch.ones_like(x)
        future_observed_mask = torch.zeros((x.shape[0], self.output_length, self.width))
        future_values = torch.zeros((x.shape[0], self.output_length, self.width))
        
        future_time_features = torch.zeros((x.shape[0], self.output_length, 1))
        for i in range(future_time_features.shape[1]):
            future_time_features[:, i, 0] = x.shape[1] + i
        
        # print(past_time_features)
        # print(future_time_features)
        
        transformer_outs = self.transformer.forward(past_values=x, past_time_features=past_time_features, past_observed_mask=past_observed_mask, future_values=future_values, future_observed_mask=future_observed_mask, future_time_features=future_time_features)
        # print("Transformer outs:", transformer_outs.encoder_last_hidden_state)
        # print(transformer_outs.decoder_hidden_states[-1].shape)
        trans_final = self.linear(transformer_outs.decoder_hidden_states[-1])

        skip_final = torch.reshape(self.skip(torch.reshape(x[:, -self.context_length:, :], (-1, self.width * self.context_length))), (-1, self.output_length, self.width))
        
        cat_finals = torch.cat((trans_final, skip_final), 2)
        # print(cat_finals.shape)
        final = self.last(cat_finals)
        
        
        return final
    
class TransformerModel2(nn.Module):

    def __init__(self, input_width, input_length, lags=False, context_length=50, num_lags = 10, output_length = 1, hidden_size=64, num_layers=2):
        super(TransformerModel2, self).__init__()

        self.name = "Transformer2"
        self.width = input_width
        self.length = input_length
        self.output_length = output_length
        
        if lags:
            self.lags_sequence = [i * (input_length - context_length)//num_lags for i in range(num_lags)]
            self.context_length = context_length
        else:
            self.lags_sequence = [1, 2, 3, 4, 5, 6, 7]
            self.context_length = input_length//2


        configuration = TimeSeriesTransformerConfig(prediction_length=output_length, 
                                                    input_size=input_width, 
                                                    context_length=self.context_length, 
                                                    num_time_features=1, 
                                                    lags_sequence = self.lags_sequence,
                                                    d_model=hidden_size, 
                                                    decoder_layers=num_layers, 
                                                    encoder_layers=num_layers, 
                                                    output_hidden_states=True)

        self.transformer = TimeSeriesTransformerForPrediction(configuration)

    def forward(self, x):
        past_time_features = torch.zeros((x.shape[0], x.shape[1], 1))
        for i in range(past_time_features.shape[1]):
            past_time_features[:, i, 0] = i

        past_observed_mask = torch.ones_like(x)
        future_observed_mask = torch.zeros((x.shape[0], self.output_length, self.width))
        future_values = torch.zeros((x.shape[0], self.output_length, self.width))
        
        future_time_features = torch.zeros((x.shape[0], self.output_length, 1))
        for i in range(future_time_features.shape[1]):
            future_time_features[:, i, 0] = x.shape[1] + i
        

        # print(x.shape)
        # print(past_time_features.shape)
        # print(future_time_features.shape)
        
        
        transformer_outs = self.transformer.generate(past_values=x, past_time_features=past_time_features, past_observed_mask=past_observed_mask, future_time_features=future_time_features)
        # print("Transformer outs:", transformer_outs.encoder_last_hidden_state)
        # print(transformer_outs.decoder_hidden_states[-1].shape)
        return transformer_outs.sequences.mean(dim=1)

    def train_forward(self, x, y):
        past_time_features = torch.zeros((x.shape[0], x.shape[1], 1))
        for i in range(past_time_features.shape[1]):
            past_time_features[:, i, 0] = i

        past_observed_mask = torch.ones_like(x)
        future_observed_mask = torch.ones((x.shape[0], self.output_length, self.width))
        
        future_time_features = torch.zeros((x.shape[0], self.output_length, 1))
        for i in range(future_time_features.shape[1]):
            future_time_features[:, i, 0] = x.shape[1] + i
        
        transformer_outs = self.transformer(past_values=x, past_time_features=past_time_features, past_observed_mask=past_observed_mask, future_observed_mask=future_observed_mask, future_time_features=future_time_features, future_values=y)
        return transformer_outs
        

        




class ANNMCModel(nn.Module):

    def __init__(self, input_width, input_length, tpm, output_length = 1, hidden_size=64, state_max=5, state_size=0.1, layer_sizes = [100, 20]):
        super(ANNMCModel, self).__init__()

        self.name = "ANNMC"
        self.width = input_width
        self.length = input_length
        self.output_length = output_length
        self.state_size = state_size
        self.state_max = state_max
        self.num_states = 2 * state_max // state_size
        self.TPM = torch.from_numpy(tpm)
        self.TPM.requires_grad = False
        self.ann1 = nn.Sequential(nn.Linear(input_width * input_length, 100), nn.ReLU(), nn.Linear(100, 25), nn.ReLU(), nn.Linear(25, input_width * output_length))
        self.ann2 = nn.Sequential(nn.Linear(input_width * 6, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, input_width))


    def forward(self, x):
        
        x_shaped = torch.reshape(x, (-1, x.shape[1] * x.shape[2]))
        first_outs = self.ann1(x_shaped)
        x2 = torch.reshape(first_outs, (-1, self.output_length, self.width))

        finals = torch.zeros((x.shape[0], self.output_length, x.shape[2]))
        for i in range(x2.shape[0]):
            for j in range(x2.shape[1]):
                ten = torch.zeros(self.width * 6)
                for k in range(x2.shape[2]):
                    if j == 0:
                        val_1 = x[i, -1, k]
                        
                    else:
                        val_1 = x2[i, j-1, k]

                    
                    
                    state_1 = (val_1 + self.state_max)/(self.state_size)
                    # print(state_1.reshape(1))
                    # print(torch.cat((state_1.reshape(1), torch.tensor([0])), dim=0))
                    state_1 = torch.round(state_1)
                    state_1 = torch.max(torch.cat((state_1.reshape(1), torch.tensor([0])), dim=0))
                    state_1 = torch.min(torch.cat((state_1.reshape(1), torch.tensor([self.num_states - 1])), dim=0))


                    val_2 = x2[i, j, k]
                    ten[k * (6)] = val_2
                    state_2 = (val_2 + self.state_max)/(self.state_size)
                    state_2 = torch.round(state_2)
                    state_2 = torch.max(torch.cat((state_2.reshape(1), torch.tensor([0])), dim=0))
                    state_2 = torch.min(torch.cat((state_2.reshape(1), torch.tensor([self.num_states - 1])), dim=0))

                    state_3 = torch.min(torch.cat((state_1.reshape(1) + 1, torch.tensor([self.num_states - 1]))))
                    state_4 = torch.max(torch.cat((state_1.reshape(1) - 1, torch.tensor([0]))))
                    state_5 = torch.min(torch.cat((state_1.reshape(1) + 2, torch.tensor([self.num_states - 1]))))
                    state_6 = torch.max(torch.cat((state_1.reshape(1) - 2, torch.tensor([0]))))

                    # print(state_1.reshape(1).int())

                    row = torch.index_select(self.TPM[k, :, :], dim=0, index=state_1.reshape(1).int())
                    ten[6*k+1] = torch.index_select(row, dim = 1, index=state_2.reshape(1).int())
                    ten[6*k+2] = torch.index_select(row, dim = 1, index=state_3.reshape(1).int())
                    ten[6*k+3] = torch.index_select(row, dim = 1, index=state_4.reshape(1).int())
                    ten[6*k+4] = torch.index_select(row, dim = 1, index=state_5.reshape(1).int())
                    ten[6*k+5] = torch.index_select(row, dim = 1, index=state_6.reshape(1).int())

                ten = torch.nan_to_num(ten)
                out = self.ann2(ten)
                # prin?t(out)
                finals[i, j, :] = out
        
        return finals