import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json

from ..data.preprocessing import load_dakshina_data, get_dataloaders
from ..models import Encoder, Decoder, Seq2Seq, AttentionDecoder, AttentionSeq2Seq
from .evaluate import evaluate, calculate_accuracy

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, iterator, optimizer, criterion, clip, device, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(iterator):
        src = batch['source'].to(device)
        trg = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        if isinstance(model, AttentionSeq2Seq):
            output, _ = model(src, trg, teacher_forcing_ratio)
        else:
            output = model(src, trg, teacher_forcing_ratio)
        
        # Exclude the first token (< SOS >) from loss calculation
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_model(config=None):
    with wandb.init(project="transliteration-seq2seq", config=config):
        config = wandb.config
        
        # Load data
        data_dict = load_dakshina_data(language=config.language, base_dir=config.data_dir)
        
        if data_dict is None:
            print("Failed to load data. Please check the dataset paths.")
            return None, 0, []
            
        train_loader, dev_loader, test_loader = get_dataloaders(
            data_dict, 
            batch_size=config.batch_size
        )
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        input_size = len(data_dict['source_vocab'])
        output_size = len(data_dict['target_vocab'])
        
        # Create encoder and decoder
        encoder = Encoder(
            input_size=input_size, 
            embedding_size=config.embedding_size, 
            hidden_size=config.hidden_size, 
            num_layers=config.encoder_layers,
            cell_type=config.cell_type,
            dropout=config.dropout
        )
        
        if config.use_attention:
            decoder = AttentionDecoder(
                output_size=output_size, 
                embedding_size=config.embedding_size, 
                encoder_hidden_size=config.hidden_size,
                decoder_hidden_size=config.hidden_size, 
                num_layers=config.decoder_layers,
                cell_type=config.cell_type,
                dropout=config.dropout
            )
            model = AttentionSeq2Seq(encoder, decoder, device).to(device)
        else:
            decoder = Decoder(
                output_size=output_size, 
                embedding_size=config.embedding_size, 
                hidden_size=config.hidden_size, 
                num_layers=config.decoder_layers,
                cell_type=config.cell_type,
                dropout=config.dropout
            )
            model = Seq2Seq(encoder, decoder, device).to(device)
        
        # Initialize weights
        model.apply(init_weights)
        
        # Print model info
        print(f'The model has {count_parameters(model):,} trainable parameters')
        
        # Define optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=data_dict['target_vocab']['<PAD>'])
        
        # Training loop
        best_valid_loss = float('inf')
        
        for epoch in range(config.n_epochs):
            start_time = time.time()
            
            train_loss = train_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                config.clip, 
                device, 
                config.teacher_forcing_ratio
            )
            
            valid_loss = evaluate(model, dev_loader, criterion, device)
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            # Calculate accuracy on validation set
            valid_accuracy, _ = calculate_accuracy(
                model, 
                dev_loader, 
                data_dict['inv_target_vocab'], 
                device
            )
            
            # Log to wandb
            wandb.log({
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_accuracy,
                "epoch": epoch
            })
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'best-model-{wandb.run.id}.pt')
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')
            print(f'\t Val. Accuracy: {valid_accuracy:.3f}')
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(f'best-model-{wandb.run.id}.pt'))
        
        # Evaluate on test set
        test_loss = evaluate(model, test_loader, criterion, device)
        test_accuracy, test_predictions = calculate_accuracy(
            model, 
            test_loader, 
            data_dict['inv_target_vocab'], 
            device
        )
        
        # Log final test metrics
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })
        
        print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f}')
        
        # Save predictions
        if config.use_attention:
            folder_name = 'predictions/attention'
        else:
            folder_name = 'predictions/vanilla'
            
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
        with open(f'{folder_name}/predictions-{wandb.run.id}.json', 'w') as f:
            json.dump(test_predictions, f, ensure_ascii=False, indent=2)
        
        # Create visualization artifacts
        if config.use_attention:
            # Generate attention visualizations for a few examples
            from ..visualization.analysis import visualize_attention
            visualize_attention(model, test_loader, data_dict, device, wandb.run.id)
        
        return model, test_accuracy, test_predictions

def run_sweep(sweep_config, count=50):
    sweep_id = wandb.sweep(sweep_config, project="transliteration-seq2seq")
    wandb.agent(sweep_id, train_model, count=count)
