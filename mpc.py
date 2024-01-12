import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from traj_opt import optimal_traj
from models import *
import torch

x0 = 0
v0 = 0
targ_x = 0
max_u = 5

dt = 0.1
t = 0
lookahead = 5
pred_horizon = 10

wind_data = pd.read_csv('sample_data.csv').to_numpy()
random.seed(12)
start_ind = int(random.randint(0, len(wind_data) - 10000))
ind = start_ind



x = x0
v = v0


xs = []
vs = []
us = []
ts = []

persistent = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow_DEG/experiments/Big_Naive_2024-01-07_11:39:07"
lin_reg = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow_DEG/experiments/Big_Lin_Reg_2024-01-08_07:51:41"
lin_reg_flat = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow_DEG/experiments/Big_Lin_Reg_Flat_2024-01-08_01:31:26"
deep_1 = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow_DEG/experiments/ Deep_Reg_2024-01-06_11:22:45"
deep_2 = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow_DEG/experiments/Big_Deep_Reg_2024-01-07_22:55:59"
deep_3 = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow_DEG/experiments/ Deep_Reg_2024-01-06_11:40:14"
rnn = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow_DEG/experiments/Big_RNN_2024-01-06_21:21:57"
lstm = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow_DEG/experiments/Big_LSTM_2024-01-07_02:21:26"
transformer = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow_DEG/experiments/Big_Transformer_2024-01-06_16:03:52"
annmc = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow_DEG/experiments/Big_ANNMC_2024-01-07_12:16:39"
paths = [persistent, lin_reg, lin_reg_flat, deep_2, rnn, lstm, transformer, annmc]

experiment_path = annmc
params_df = pd.read_csv(experiment_path + '/params.csv')
input_length = params_df['input_length'].iloc[0]
output_len = params_df['output_length'].iloc[0]
input_width = params_df['input_width'].iloc[0]
hidden_size = params_df['hidden_size'].iloc[0]
num_layers = params_df['num_layers'].iloc[0]
input_cols = [0 for i in range(input_width)]
tpm = np.load('tpm.npy')

my_model = NaiveModel(len(input_cols), input_length, output_length=output_len)
# my_model = RecurrentModel(len(input_cols), input_length, output_length=output_len, hidden_size=hidden_size, num_layers=num_layers)
# my_model = LinearRegression(len(input_cols), input_length, output_length=output_len)
# my_model = ANNMCModel(len(input_cols), input_length, tpm, output_length=output_len)
# my_model = LSTMModel(len(input_cols), input_length, output_length=output_len, hidden_size=hidden_size, num_layers=num_layers)
# my_model = TransformerModel(len(input_cols), input_length, output_length=output_len, hidden_size=hidden_size, num_layers=num_layers, lags=True)
# my_model = LinearRegressionFlat(len(input_cols), input_length, output_length=output_len)
# my_model = DeepRegression(len(input_cols), input_length, output_length=output_len, layer_sizes=[50, 50])

if my_model.name != "Naive":
    my_model.load_state_dict(torch.load(experiment_path + '/weights.pt'))



while t < 100:

    prev_wind = wind_data[ind-1, 0]
    # print(prev_wind)

    prev_winds = wind_data[ind-input_length:ind, :]
    # print(prev_winds)
    prev_winds = torch.tensor(prev_winds)
    prev_winds = prev_winds.float()
    prev_winds = prev_winds.reshape((1, input_length, input_width))
    # print(prev_winds)
    # print(prev_winds.shape)

    next_winds = my_model(prev_winds)
    next_wind = next_winds.detach().numpy()[0, :, 0]
    # print(prev_wind, next_wind)
    pred_winds = next_wind
    # pred_winds = [next_wind for i in range(pred_horizon + 1)]
    # X = optimal_traj(targ_x, x, v, pred_horizon, dt, max_u)
    # X = optimal_traj(targ_x, x, v, pred_horizon, dt, max_u, winds=pred_winds)
    X = optimal_traj(targ_x, x, v, pred_horizon, dt, max_u, winds=wind_data[ind:ind+pred_horizon+1, 0])
    new_us = X[2*(pred_horizon+1):2*(pred_horizon+1)+lookahead]
    new_fs = new_us + wind_data[ind:ind+lookahead, 0]

    for i in range(lookahead):
        new_u = X[2*(pred_horizon+1) + i]
        new_f = new_u + wind_data[ind + i, 0]
        new_v = v + new_f * dt
        new_x = x + v * dt

        ts.append(t)
        xs.append(x)
        vs.append(v)
        us.append(new_u)

        t = t + dt
        v = new_v
        x = new_x

    ind += lookahead

fig, axs = plt.subplots(2)
axs[0].plot(ts, xs)
axs[0].set_ylabel('Drone Displacement')
axs[1].plot(wind_data[start_ind:start_ind+int(t//dt)])
axs[1].set_ylabel('Wind Speed')
axs[1].set_xlabel('Time')
axs[0].set_title('Control with DNN Wind Predictions')
plt.show()
print(np.sqrt(np.mean(np.square(xs))))

d = {'t': ts, 'x': xs}
df = pd.DataFrame.from_dict(d)
# df.to_csv(f"{my_model.name}_sim_results.csv", index=False)
df.to_csv("Optimal_sim_results.csv", index=False)