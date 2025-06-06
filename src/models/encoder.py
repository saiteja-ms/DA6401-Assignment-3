import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, cell_type='lstm', dropout=0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_size, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                embedding_size, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # vanilla RNN
            self.rnn = nn.RNN(
                embedding_size, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                nonlinearity='tanh'  # Explicitly set to tanh for vanilla RNN
            )
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, seq_length, embedding_size)
        
        if self.cell_type == 'lstm':
            outputs, (hidden, cell) = self.rnn(embedded)
            return outputs, hidden, cell
        else:
            outputs, hidden = self.rnn(embedded)
            return outputs, hidden, None
