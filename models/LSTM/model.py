import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size = 2,dropout_rate=0.2):
        super(LSTM,self).__init__()

        self.lstm = nn.LSTM(
            input_size= input_size,
            hidden_size= hidden_size,
            num_layers= num_layers,
            batch_first= True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size,output_size)

    # forward pass
    def forward(self, x):
        """ 
        output is the full set of hidden states
        hh is last hidden state
        cn is the last cell state
        """
        out, (hh,cn) = self.lstm(x)
        out = out[:,-1,:] # extract the last timestep
        out = self.dropout(out) # dropout
        out = self.fc(out) # linear prediction
                  
        return out

