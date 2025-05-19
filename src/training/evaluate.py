import torch
import torch.nn as nn
from tqdm import tqdm
from ..models import AttentionSeq2Seq

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in iterator:
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            
            if isinstance(model, AttentionSeq2Seq):
                output, _ = model(src, trg, 0)  # No teacher forcing during evaluation
            else:
                output = model(src, trg, 0)  # No teacher forcing during evaluation
            
            # Exclude the first token (< SOS >) from loss calculation
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def calculate_accuracy(model, iterator, inv_target_vocab, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    # Find the EOS token index
    eos_idx = None
    if '<EOS>' in inv_target_vocab:
        eos_idx = inv_target_vocab['<EOS>']
    else:
        # If not found directly, try to get it from the value
        for idx, token in inv_target_vocab.items():
            if token == '<EOS>':
                eos_idx = idx
                break
    
    # If still not found, use the default index (usually 3)
    if eos_idx is None:
        eos_idx = 3
    
    # Find the PAD token index
    pad_idx = None
    if '<PAD>' in inv_target_vocab:
        pad_idx = inv_target_vocab['<PAD>']
    else:
        # If not found directly, try to get it from the value
        for idx, token in inv_target_vocab.items():
            if token == '<PAD>':
                pad_idx = idx
                break
    
    # If still not found, use the default index (usually 0)
    if pad_idx is None:
        pad_idx = 0
    
    # Find the UNK token index
    unk_idx = None
    if '<UNK>' in inv_target_vocab:
        unk_idx = inv_target_vocab['<UNK>']
    else:
        # If not found directly, try to get it from the value
        for idx, token in inv_target_vocab.items():
            if token == '<UNK>':
                unk_idx = idx
                break
    
    # If still not found, use the default index (usually 1)
    if unk_idx is None:
        unk_idx = 1
    
    with torch.no_grad():
        for batch in iterator:
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            src_texts = batch['source_text']
            trg_texts = batch['target_text']
            
            batch_size = src.shape[0]
            
            # Initialize with < SOS > token
            decoder_input = trg[:, 0].unsqueeze(1)
            
            # For storing decoded outputs
            decoded_outputs = torch.zeros(batch_size, trg.shape[1], dtype=torch.long).to(device)
            decoded_outputs[:, 0] = decoder_input.squeeze(1)
            
            if isinstance(model, AttentionSeq2Seq):
                encoder_outputs, hidden, cell = model.encoder(src)
                if model.cell_type != 'lstm':
                    cell = None
                
                for t in range(1, trg.shape[1]):
                    if model.cell_type == 'lstm':
                        decoder_output, hidden, cell, _ = model.decoder(
                            decoded_outputs[:, t-1], hidden, encoder_outputs, cell
                        )
                    else:
                        decoder_output, hidden, _, _ = model.decoder(
                            decoded_outputs[:, t-1], hidden, encoder_outputs
                        )
                    
                    top1 = decoder_output.argmax(1)
                    decoded_outputs[:, t] = top1
            else:
                if model.cell_type == 'lstm':
                    encoder_outputs, hidden, cell = model.encoder(src)
                else:
                    encoder_outputs, hidden, _ = model.encoder(src)
                    cell = None
                
                for t in range(1, trg.shape[1]):
                    if model.cell_type == 'lstm':
                        decoder_output, hidden, cell = model.decoder(
                            decoded_outputs[:, t-1], hidden, cell
                        )
                    else:
                        decoder_output, hidden, _ = model.decoder(
                            decoded_outputs[:, t-1], hidden
                        )
                    
                    top1 = decoder_output.argmax(1)
                    decoded_outputs[:, t] = top1
            
            # Convert indices to characters
            for i in range(batch_size):
                pred_indices = decoded_outputs[i, 1:].tolist()  # Skip < SOS >
                pred_chars = []
                
                for idx in pred_indices:
                    # Use the indices we found earlier
                    if idx == eos_idx:
                        break
                    if idx != pad_idx and idx != unk_idx:
                        pred_chars.append(inv_target_vocab[idx])
                
                pred_text = ''.join(pred_chars)
                
                if pred_text == trg_texts[i]:
                    correct += 1
                
                predictions.append({
                    'source': src_texts[i],
                    'target': trg_texts[i],
                    'prediction': pred_text,
                    'correct': pred_text == trg_texts[i]
                })
                
                total += 1
    
    accuracy = correct / total
    return accuracy, predictions
