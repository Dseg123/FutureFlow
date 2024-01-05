import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from traj_opt_2d import optimal_traj
import models
import torch

x0 = 0
y0 = 0
vx0 = 0
vy0 = 0
targ_x = 0
targ_y = 0
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
vx = vx0
y = y0
vy = vy0

xs = []
ys = []
vxs = []
vys = []
uxs = []
uys = []
ts = []

# experiment_path = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow/experiments/Transformer2_2023-12-10_19:26:44"
# experiment_path = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow/experiments/Lin_Reg_Flat_2023-12-10_14:52:04"
experiment_path = "/home/dylaneg/Documents/Programming/IROM/FutureFlow/FutureFlow/experiments/Deep_Reg_2023-12-10_20:33:15"
params_df = pd.read_csv(experiment_path + '/params.csv')
input_length = params_df['input_length'].iloc[0]
output_length = params_df['output_length'].iloc[0]
input_width = params_df['input_width'].iloc[0]
hidden_size = params_df['hidden_size'].iloc[0]
num_layers = params_df['num_layers'].iloc[0]
model = models.DeepRegression(input_length = input_length, input_width = input_width, output_length = output_length)
# model = models.LinearRegressionFlat(input_length = input_length, input_width = input_width, output_length = output_length)
# model = models.TransformerModel2(input_length = input_length, input_width = input_width, output_length = output_length, hidden_size=hidden_size, num_layers=num_layers)
model.load_state_dict(torch.load(experiment_path + '/weights.pt'))



while t < 100:

    prev_wind = wind_data[ind-1, 0]
    # print(prev_wind)

    prev_winds = wind_data[ind-input_length:ind, :]
    # print(prev_winds)
    prev_winds = torch.tensor(prev_winds)
    prev_winds = prev_winds.float()
    prev_winds = prev_winds.reshape((1, input_length, input_width))
    # print(prev_winds)
    next_winds = model(prev_winds)
    next_wind = next_winds.detach().numpy()[0, :, :2]
    # print(prev_wind, next_wind)
    pred_winds = next_wind
    # pred_winds = [next_wind for i in range(pred_horizon + 1)]
    naive_winds = [prev_wind for i in range(pred_horizon + 1)]
    # X = optimal_traj(targ_x, targ_y, x, y, vx, vy, pred_horizon, dt, max_u)
    # X = optimal_traj(targ_x, x, v, pred_horizon, dt, max_u, winds=naive_winds)
    # X = optimal_traj(targ_x, targ_y, x, y, vx, vy, pred_horizon, dt, max_u, winds=pred_winds)
    X = optimal_traj(targ_x, targ_y, x, y, vx, vy, pred_horizon, dt, max_u, winds=wind_data[ind:ind+pred_horizon+1, :2])
    new_uxs = X[4*(pred_horizon+1):4*(pred_horizon+1)+lookahead]
    new_uys = X[5*(pred_horizon+1):5*(pred_horizon+1) + lookahead]
    new_fxs = new_uxs + wind_data[ind:ind+lookahead, 0]
    new_fys = new_uys + wind_data[ind:ind+lookahead, 1]

    for i in range(lookahead):
        new_ux = X[4*(pred_horizon+1) + i]
        new_fx = new_ux + wind_data[ind + i, 0]
        new_vx = vx + new_fx * dt
        new_x = x + vx * dt

        ts.append(t)
        xs.append(x)
        vxs.append(vx)
        uxs.append(new_ux)

        new_uy = X[5*(pred_horizon+1) + i]
        new_fy = new_uy + wind_data[ind + i, 1]
        new_vy = vy + new_fy * dt
        new_y = y + vy * dt

        ys.append(y)
        vys.append(vy)
        uys.append(new_uy)



        t = t + dt
        vx = new_vx
        vy = new_vy
        x = new_x
        y = new_y

    ind += lookahead


plt.plot(ts, ys)
plt.show()
# fig, axs = plt.subplots(2)
# axs[0].plot(ts, xs)
# axs[0].plot()
# axs[0].set_ylabel('Drone Displacement')
# axs[1].plot(wind_data[start_ind:start_ind+int(t//dt)])
# axs[1].set_ylabel('Wind Speed')
# axs[1].set_xlabel('Time')
# axs[0].set_title('Control with DNN Wind Predictions')
# plt.show()
# print(np.sqrt(np.mean(np.square(xs))))

# d = {'t': ts, 'x': xs}
# df = pd.DataFrame.from_dict(d)
# df.to_csv("Transformer_sim_results.csv", index=False)