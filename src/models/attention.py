import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, decoder_hidden_dim]
        # encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        # Get attention weights
        return F.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, encoder_hidden_size, decoder_hidden_size, 
                 num_layers=1, cell_type='lstm', dropout=0):
        super(AttentionDecoder, self).__init__()
        self.output_size = output_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attention = Attention(encoder_hidden_size, decoder_hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Input to the RNN will be embedding + context vector
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_size + encoder_hidden_size, 
                decoder_hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                embedding_size + encoder_hidden_size, 
                decoder_hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # vanilla RNN
            self.rnn = nn.RNN(
                embedding_size + encoder_hidden_size, 
                decoder_hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
        self.fc = nn.Linear(decoder_hidden_size + encoder_hidden_size + embedding_size, output_size)
    
    def forward(self, x, hidden, encoder_outputs, cell=None):
        # x shape: (batch_size)
        # hidden shape: (num_layers, batch_size, decoder_hidden_size)
        # encoder_outputs shape: (batch_size, src_len, encoder_hidden_size)
        
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(x))  # (batch_size, 1, embedding_size)
        
        # Get the last hidden state for attention
        if self.cell_type == 'lstm':
            attn_hidden = hidden[-1]
        else:
            attn_hidden = hidden[-1]
            
        # Calculate attention weights
        attn_weights = self.attention(attn_hidden, encoder_outputs)  # (batch_size, src_len)
        
        # Create context vector by multiplying attention weights with encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, encoder_hidden_size)
        
        # Combine embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embedding_size + encoder_hidden_size)
        
        # Pass through RNN
        if self.cell_type == 'lstm':
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        else:
            output, hidden = self.rnn(rnn_input, hidden)
            cell = None
            
        # Final output layer
        # Concatenate output, context and embedded for richer representation
        output = output.squeeze(1)  # (batch_size, decoder_hidden_size)
        context = context.squeeze(1)  # (batch_size, encoder_hidden_size)
        embedded = embedded.squeeze(1)  # (batch_size, embedding_size)
        
        prediction = self.fc(torch.cat((output, context, embedded), dim=1))
        
        return prediction, hidden, cell, attn_weights
