import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers=1, cell_type='lstm', dropout=0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        self.embedding = nn.Embedding(output_size, embedding_size)
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
            
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell=None):
        # x shape: (batch_size, 1)
        x = x.unsqueeze(1)
        
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, 1, embedding_size)
        
        if self.cell_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            prediction = self.fc(output.squeeze(1))
            return prediction, hidden, cell
        else:
            output, hidden = self.rnn(embedded, hidden)
            prediction = self.fc(output.squeeze(1))
            return prediction, hidden, None
