# FutureFlow

A Deep Learning Approach to Short-Term Wind Gust Prediction and Control
Dylan Epstein-Gross

This repository contains the code, visualizations, and experiment results
used in my Fall 2023 Junior Independent Work Project with Dr. Anirudha
Majumdar and Nathaniel Simon in Princton's Intelligent Robot Motion Lab.

For more information about the project, you can check out my paper at
https://drive.google.com/file/d/1UAizXDJcXc1AKrbifOOvQkIRZd1BkK7e/view?usp=drive_link, slides at https://docs.google.com/presentation/d/1xeKvwHW5VJzHZ14dohGbVrJD91OO7VNrMZrJekbB5BA/edit?usp=sharing, and oral presentation at https://drive.google.com/file/d/1zk3DnDJdHnHRmSKvHUSEOcaD93ElZ4KK/view?usp=sharing.

If you want to explore this repository yourself, here is some advice for
navigating it.

## Code

The code in this repo consists primarily of .py files used for final algorithms
and .ipynb files used for testing things out (so these can largely be ignored).
The following are some important files:

dataset.py -- defines the Pytorch dataset used to train the models

losses.py -- defines the loss functions used to train the models

models.py -- defines the architectures of each of the models used in the paper

markov.py -- computes the Markov transition matrix used in the ANN-MC model

train_model.py -- creates and trains one of the models on data of your choice, and
stores the experiment result

test_model.py -- tests a pre-trained model on data of your choice, outputting the loss

traj_opt.py -- OSQP trajectory optimization used in simulation

mpc.py -- Runs a single MPC simulation using a model of your choice, and stores the
result

multimpc.py -- Runs many MPC simulations and averages the resulting errors

visualizations.py -- helper functions for visualizing model predictions

von_karman.py -- synthesizes von Karman data as described in the paper

animate_sim.py -- animates simulation results and stores as mp4

The vis\_\*.ipynb files contain the code used to produce all the visualizations in the paper,
so may be worth looking at also.

## Data

Raw data are stored as .mat files in the /datasets directory. Training data is in sample_data.csv
and testing data is in test_data.csv. The synthesized train and test data are in von_karman.csv.
The other datasets were synthesized to test other aspects of the models, such as a simple sine wave
in sine_data.csv and a down-sampled function in freq_20_data.csv.

## Simulation Results

Simulation results are stored as .csv files in the /sim_results directory. They simply consist
of the time t and position x at that time for the simulation. Videos of these simulations can
also be found in the /vids directory.

## Experiments

The results of model training are stored as separate folders in the /experiments directory. Each
folder consists of a log.csv tracking the progress of the model (epoch, train loss, and test loss), a params.csv storing the hyperparameters used in the model (learning rate, layer sizes, etc.) and a weights.pt storing the weights of the model for later testing. The title of the folder
includes the name of the model, the timestamp at which it was trained, and in some cases a string
communicating the dataset it was trained on (such as "VK" for von Karman dataset).
