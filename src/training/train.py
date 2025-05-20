import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
import random

from ..data.preprocessing import load_dakshina_data, get_dataloaders
from ..models import Encoder, Decoder, Seq2Seq, AttentionDecoder, AttentionSeq2Seq
from .evaluate import evaluate, calculate_accuracy

def init_weights(m):
    """Initializes model weights."""
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model: nn.Module, iterator: torch.utils.data.DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, clip: float, device: torch.device, teacher_forcing_ratio: float) -> float:
    """
    Trains the model for one epoch.
    Uses attestation counts as weights for the loss function.
    """
    model.train()
    epoch_loss = 0

    for batch in tqdm(iterator, desc="Training"):
        src = batch['source'].to(device)
        trg = batch['target'].to(device)
        attestation = batch['attestation'].to(device)  # Get attestation counts
        
        optimizer.zero_grad()
        
        if isinstance(model, AttentionSeq2Seq):
            output, _ = model(src, trg, teacher_forcing_ratio)
        else:
            output = model(src, trg, teacher_forcing_ratio)

        output_dim = output.shape[-1]

        # Slice the output and target to remove the first timestep (which corresponds to the < SOS > prediction)
        output_seq = output[:, 1:, :]  # Shape [batch_size, trg_len-1, output_dim]
        trg_seq = trg[:, 1:]           # Shape [batch_size, trg_len-1]
        
        # Check for empty sequences after slicing (can happen with max_len=1 or 2 or small batches)
        if trg_seq.numel() == 0:
            continue  # Skip batch if no valid target tokens remain

        # Calculate unweighted loss
        loss = criterion(output_seq.reshape(-1, output_dim), trg_seq.reshape(-1))
        
        # Apply attestation weights to the loss
        # Reshape attestation to match the batch elements
        seq_len = trg_seq.shape[1]  # Length after removing SOS
        attestation_weights = attestation.repeat_interleave(seq_len)
        
        # Normalize weights to sum to batch size
        attestation_weights = attestation_weights * (attestation_weights.size(0) / attestation_weights.sum())
        
        # Apply weights to loss
        weighted_loss = (loss * attestation_weights).mean()
        
        weighted_loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += weighted_loss.item()

    # Handle case where iterator was empty or all batches were skipped
    return epoch_loss / len(iterator) if len(iterator) > 0 else 0.0

def epoch_time(start_time: float, end_time: float) -> tuple[int, int]:
    """Calculates elapsed time in minutes and seconds."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_model(config=None):
    """
    Main training function with descriptive run names.
    """
    # Generate a descriptive run name
    run_name = None
    if config:
        # Model type
        model_type = "attention" if getattr(config, 'use_attention', False) else "vanilla"
        
        # Cell type
        cell_type = getattr(config, 'cell_type', 'lstm')
        
        # Architecture details
        emb_size = getattr(config, 'embedding_size', 64)
        hid_size = getattr(config, 'hidden_size', 128)
        
        # Layer information
        enc_layers = getattr(config, 'encoder_layers', 1)
        dec_layers = getattr(config, 'decoder_layers', 1)
        
        # Training parameters
        optimizer_name = getattr(config, 'optimizer', 'adam')
        lr = getattr(config, 'learning_rate', 0.001)
        dropout = getattr(config, 'dropout', 0.0)
        batch_size = getattr(config, 'batch_size', 64)
        
        # Create meaningful run name
        run_name = f"{model_type}_{cell_type}_emb{emb_size}_hid{hid_size}_enc{enc_layers}_dec{dec_layers}_drop{dropout}_{optimizer_name}_lr{lr:.6f}_batch{batch_size}"
        print(f"Generated run name: {run_name}")
    
    # Set thread start method for wandb
    os.environ["WANDB_START_METHOD"] = "thread"
    
    # Initialize wandb with the descriptive run name - IMPORTANT: Don't use with statement
    wandb_run = wandb.init(project="transliteration-seq2seq", config=config, name=run_name, settings=wandb.Settings(start_method="thread"))
    print(f"Actual wandb run name: {wandb_run.name}")
    
    if run_name and wandb.run:
        wandb.run.name = run_name
        wandb.run.save()

    print(f"Actual wandb run name: {wandb.run.name}")
    
    config = wandb.config

    print(f"Starting training run with config: {config}")

    # --- Data Loading ---
    data_dict = load_dakshina_data(
        language=config.language,
        base_dir=config.data_dir,
        max_len=getattr(config, 'max_seq_len', 50)
    )

    # Handle potential data loading failure
    if data_dict is None or not data_dict['train_dataset'] or len(data_dict['train_dataset']) < config.batch_size:
        print("Failed to load data or train dataset is too small. Exiting training.")
        if wandb.run:
             wandb.log({"train_loss": float('nan'), "valid_loss": float('nan'), "valid_accuracy": 0.0, "test_accuracy": 0.0})
             wandb.run.finish(exit_code=1)
        return None, 0, []

    train_loader, dev_loader, test_loader = get_dataloaders(
        data_dict,
        batch_size=config.batch_size
    )

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Model Creation ---
    input_size = len(data_dict['source_vocab'])
    output_size = len(data_dict['target_vocab'])
    pad_idx = data_dict['target_vocab'].get('<PAD>', 0)

    # Determine number of layers
    num_layers = getattr(config, 'num_layers', 1)
    num_encoder_layers = num_layers
    num_decoder_layers = num_layers


    # Create encoder
    encoder = Encoder(
        input_size=input_size,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_layers=num_encoder_layers,
        cell_type=config.cell_type,
        dropout=config.dropout
    ).to(device)

    # Create decoder (Attention or Vanilla)
    if config.use_attention:
        decoder = AttentionDecoder(
            output_size=output_size,
            embedding_size=config.embedding_size,
            encoder_hidden_size=config.hidden_size,
            decoder_hidden_size=config.hidden_size,
            num_layers=num_decoder_layers,
            cell_type=config.cell_type,
            dropout=config.dropout
        ).to(device)
        model = AttentionSeq2Seq(encoder, decoder, device).to(device)
    else:
        decoder = Decoder(
            output_size=output_size,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=num_decoder_layers,
            cell_type=config.cell_type,
            dropout=config.dropout
        ).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)

    # Initialize weights
    model.apply(init_weights)

    # Print model info
    print(f'The model has {count_parameters(model):,} trainable parameters')
    wandb.log({"trainable_parameters": count_parameters(model)})

    # Define optimizer and criterion
    optimizer_name = getattr(config, 'optimizer', 'adam').lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=getattr(config, 'weight_decay', 0)
        )
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(), 
            lr=config.learning_rate,
            alpha=getattr(config, 'rmsprop_alpha', 0.99),
            eps=getattr(config, 'rmsprop_eps', 1e-8),
            weight_decay=getattr(config, 'weight_decay', 0)
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config.learning_rate,
            momentum=getattr(config, 'momentum', 0),
            weight_decay=getattr(config, 'weight_decay', 0)
        )
    else:
        print(f"Warning: Unknown optimizer '{optimizer_name}'. Using Adam.")
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Log the optimizer type
    wandb.log({"optimizer_type": optimizer_name})
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none')  # Use 'none' to apply attestation weights

    # --- Learning Rate Scheduler ---
    scheduler_name = getattr(config, 'scheduler', 'none').lower()
    scheduler = None
    
    if scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=getattr(config, 'scheduler_factor', 0.1),
            patience=getattr(config, 'scheduler_patience', 10),
            verbose=True
        )
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=getattr(config, 'scheduler_t_max', config.n_epochs),
            eta_min=getattr(config, 'scheduler_eta_min', 0)
        )
    elif scheduler_name != 'none':
        print(f"Warning: Unknown scheduler '{scheduler_name}'. Not using a scheduler.")

    # --- Training Loop ---
    best_valid_loss = float('inf')
    best_valid_accuracy = 0.0

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

        # Evaluate on development set
        valid_loss = evaluate(model, dev_loader, criterion, device)

        # Calculate accuracy on validation set
        valid_accuracy, _ = calculate_accuracy(
            model,
            dev_loader,
            data_dict['inv_target_vocab'],
            device
        )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Step the scheduler if using one
        if scheduler is not None:
            if scheduler_name == 'plateau':
                scheduler.step(valid_loss)
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
            "epoch": epoch,
            "epoch_time_min": epoch_mins,
            "epoch_time_sec": epoch_secs,
            "learning_rate": current_lr
        })

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f}')
        print(f'\t Val. Accuracy: {valid_accuracy:.4f}')
        print(f'\t Learning Rate: {current_lr:.6f}')

        # Save the best model based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_accuracy = valid_accuracy
            model_save_path = f'best-model-{wandb.run.id}.pt'
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model to {model_save_path}")

    print("Training finished.")

    # --- Final Evaluation on Test Set ---
    print("Evaluating best model on test set...")
    model_save_path = f'best-model-{wandb.run.id}.pt'
    if os.path.exists(model_save_path):
         model.to(device)
         model.load_state_dict(torch.load(model_save_path, map_location=device))
         model.eval()

         # Calculate test loss
         test_loss = evaluate(model, test_loader, criterion, device)

         # Calculate test accuracy and get predictions
         test_accuracy, test_predictions = calculate_accuracy(
             model,
             test_loader,
             data_dict['inv_target_vocab'],
             device
         )

         # Log final test metrics
         if wandb.run:
             wandb.log({
                 "test_loss": test_loss,
                 "test_accuracy": test_accuracy
             })
             print(f'Test Loss: {test_loss:.4f}')
             print(f'Test Accuracy: {test_accuracy:.4f}')

             # --- Save Predictions ---
             folder_name = 'predictions/vanilla'
             if config.use_attention:
                 folder_name = 'predictions/attention'

             os.makedirs(folder_name, exist_ok=True)

             predictions_file_path = f'{folder_name}/predictions-{wandb.run.id}.json'
             with open(predictions_file_path, 'w', encoding='utf-8') as f:
                 # Include attestation counts in the saved predictions
                 serializable_predictions = []
                 for i, p in enumerate(test_predictions):
                      serializable_prediction = {
                           'source': p['source'],
                           'target': p['target'],
                           'prediction': p['prediction'],
                           'correct': bool(p['correct']),
                           'attestation': float(p['attestation'])
                      }
                      
                      # Include attention weights if available
                      if 'attention_weights' in p:
                          serializable_prediction['attention_weights'] = [weights.tolist() for weights in p['attention_weights']]
                          
                      serializable_predictions.append(serializable_prediction)
                      
                 json.dump(serializable_predictions, f, ensure_ascii=False, indent=2)
             print(f"Saved predictions to {predictions_file_path}")

             # --- Create Visualization Artifacts (if Attention) ---
             if isinstance(model, AttentionSeq2Seq) and len(test_loader.dataset) > 0:
                  print("Generating attention visualizations...")
                  try:
                       from ..visualization.analysis import visualize_attention, visualize_connectivity
                       visualize_attention(model, test_loader, data_dict, device, wandb.run.id)
                       visualize_connectivity(model, test_loader, data_dict, device, wandb.run.id)
                  except Exception as e:
                       print(f"Error generating visualizations: {e}")
                       import traceback
                       traceback.print_exc()
             else:
                  if config.use_attention:
                       print("Skipping attention visualizations (model type is not AttentionSeq2Seq or test dataset is empty).")

             wandb.run.finish()
         return model, test_accuracy, test_predictions
    else:
         print(f"Best model file {model_save_path} not found. Cannot perform test evaluation.")
         if wandb.run and wandb.run.state == 'running':
              if wandb.run.summary.get("test_accuracy") is None:
                  wandb.log({"test_loss": float('nan'), "test_accuracy": 0.0})
              wandb.run.finish(exit_code=1)
         return None, 0, []

def run_sweep(sweep_config: dict, count: int = 50):
    """
    Runs a wandb hyperparameter sweep.
    """
    # Set thread start method for wandb
    os.environ["WANDB_START_METHOD"] = "thread"
    
    # Add project name to sweep_config
    sweep_config['project'] = "transliteration-seq2seq"
    
    print(f"Creating sweep with configuration: {sweep_config['method']}")
    sweep_id = wandb.sweep(
        sweep_config
    )
    print(f"Starting sweep with ID: {sweep_id}")
    print(f"Running {count} trials.")
    wandb.agent(sweep_id, train_model, count=count)
