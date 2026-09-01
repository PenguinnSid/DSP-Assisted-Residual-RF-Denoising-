import torch
import torch.nn as nn
import torch.optim as optim

from model import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparams

input_size = 2
hidden_size = 64 # number of neurons/memory cells
num_layers = 1 # number of LSTMS
output_size = 2 # final dimension of prediction
learning_rate = 0.01
#droput_rate = 0.2 # (only if num_layers > 1) how many neurons will be deactivated, avoids overfitting
epochs = 10


# initialize model
model = LSTM(input_size,hidden_size,num_layers,output_size,dropout_rate = 0).to(device)
