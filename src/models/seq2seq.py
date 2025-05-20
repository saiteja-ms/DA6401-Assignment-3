import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.cell_type = encoder.cell_type
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        if self.cell_type == 'lstm':
            encoder_outputs, hidden, cell = self.encoder(source)
            
            # Adjust hidden and cell dimensions if encoder and decoder layers differ
            if self.encoder.num_layers != self.decoder.num_layers:
                # If encoder has fewer layers than decoder, repeat the layers
                if self.encoder.num_layers < self.decoder.num_layers:
                    repeat_factor = self.decoder.num_layers // self.encoder.num_layers
                    hidden = hidden.repeat(repeat_factor, 1, 1)
                    cell = cell.repeat(repeat_factor, 1, 1)
                # If encoder has more layers than decoder, take the last layers
                else:
                    hidden = hidden[-self.decoder.num_layers:]
                    cell = cell[-self.decoder.num_layers:]
        else:
            encoder_outputs, hidden, _ = self.encoder(source)
            cell = None
            
            # Adjust hidden dimensions if encoder and decoder layers differ
            if self.encoder.num_layers != self.decoder.num_layers:
                # If encoder has fewer layers than decoder, repeat the layers
                if self.encoder.num_layers < self.decoder.num_layers:
                    repeat_factor = self.decoder.num_layers // self.encoder.num_layers
                    hidden = hidden.repeat(repeat_factor, 1, 1)
                # If encoder has more layers than decoder, take the last layers
                else:
                    hidden = hidden[-self.decoder.num_layers:]
        
        # First input to the decoder is the <SOS> token
        decoder_input = target[:, 0]
        
        for t in range(1, target_len):
            if self.cell_type == 'lstm':
                decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            else:
                decoder_output, hidden, _ = self.decoder(decoder_input, hidden)
                
            outputs[:, t, :] = decoder_output
            
            # Teacher forcing: use actual target as next input
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1
        
        return outputs
