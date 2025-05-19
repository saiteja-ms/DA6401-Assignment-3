import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import wandb
from collections import Counter
import os

def analyze_errors(predictions_file):
    """Analyze errors in the predictions"""
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(predictions)
    
    # Calculate overall accuracy
    accuracy = df['correct'].mean()
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Analyze errors by word length
    df['source_len'] = df['source'].apply(len)
    df['target_len'] = df['target'].apply(len)
    
    # Group by source length
    length_accuracy = df.groupby('source_len')['correct'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(length_accuracy['source_len'], length_accuracy['correct'])
    plt.xlabel('Source Word Length')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Source Word Length')
    plt.savefig('accuracy_by_length.png')
    wandb.log({"accuracy_by_length": wandb.Image(plt)})
    
    # Analyze character-level errors
    error_pairs = []
    
    for _, row in df[~df['correct']].iterrows():
        target = row['target']
        pred = row['prediction']
        
        # Align the strings for comparison
        min_len = min(len(target), len(pred))
        
        for i in range(min_len):
            if target[i] != pred[i]:
                error_pairs.append((target[i], pred[i]))
    
    # Count error pairs
    error_counts = Counter(error_pairs)
    most_common_errors = error_counts.most_common(10)
    
    print("Most common character-level errors:")
    for (target, pred), count in most_common_errors:
        print(f"  '{target}' predicted as '{pred}': {count} times")
    
    # Create confusion matrix for most common characters
    all_target_chars = ''.join(df['target'])
    char_counts = Counter(all_target_chars)
    most_common_chars = [char for char, _ in char_counts.most_common(20)]
    
    # Filter error pairs to only include most common characters
    filtered_errors = [(t, p) for (t, p) in error_pairs if t in most_common_chars]
    
    # Create labels and counts for confusion matrix
    labels = sorted(set([t for (t, _) in filtered_errors] + [p for (_, p) in filtered_errors]))
    
    # Initialize confusion matrix
    cm = np.zeros((len(labels), len(labels)))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Fill confusion matrix
    for (t, p) in filtered_errors:
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Character Confusion Matrix')
    plt.savefig('character_confusion_matrix.png')
    wandb.log({"character_confusion_matrix": wandb.Image(plt)})
    
    # Create a sample grid of predictions
    correct_samples = df[df['correct']].sample(min(5, sum(df['correct']))).to_dict('records')
    incorrect_samples = df[~df['correct']].sample(min(5, sum(~df['correct']))).to_dict('records')
    
    samples = correct_samples + incorrect_samples
    
    # Create a table for visualization
    table_data = []
    for sample in samples:
        table_data.append([
            sample['source'],
            sample['target'],
            sample['prediction'],
            '✓' if sample['correct'] else '✗'
        ])
    
    table = wandb.Table(data=table_data, columns=["Source", "Target", "Prediction", "Correct"])
    wandb.log({"prediction_samples": table})
    
    return accuracy, most_common_errors

def compare_models(vanilla_predictions_file, attention_predictions_file):
    """Compare vanilla and attention models"""
    with open(vanilla_predictions_file, 'r') as f:
        vanilla_predictions = json.load(f)
    
    with open(attention_predictions_file, 'r') as f:
        attention_predictions = json.load(f)
    
    # Convert to DataFrames
    vanilla_df = pd.DataFrame(vanilla_predictions)
    attention_df = pd.DataFrame(attention_predictions)
    
    # Ensure the order is the same
    vanilla_df = vanilla_df.sort_values('source').reset_index(drop=True)
    attention_df = attention_df.sort_values('source').reset_index(drop=True)
    
    # Calculate accuracies
    vanilla_accuracy = vanilla_df['correct'].mean()
    attention_accuracy = attention_df['correct'].mean()
    
    print(f"Vanilla model accuracy: {vanilla_accuracy:.4f}")
    print(f"Attention model accuracy: {attention_accuracy:.4f}")
    
    # Find examples where attention model corrected vanilla model's errors
    corrected_examples = []
    
    for i in range(len(vanilla_df)):
        if not vanilla_df.iloc[i]['correct'] and attention_df.iloc[i]['correct']:
            corrected_examples.append({
                'source': vanilla_df.iloc[i]['source'],
                'target': vanilla_df.iloc[i]['target'],
                'vanilla_prediction': vanilla_df.iloc[i]['prediction'],
                'attention_prediction': attention_df.iloc[i]['prediction']
            })
    
    print(f"Number of examples corrected by attention model: {len(corrected_examples)}")
    
    # Create a table of corrected examples
    if corrected_examples:
        table_data = []
        for example in corrected_examples[:10]:  # Show top 10 examples
            table_data.append([
                example['source'],
                example['target'],
                example['vanilla_prediction'],
                example['attention_prediction']
            ])
        
        table = wandb.Table(data=table_data, columns=[
            "Source", "Target", "Vanilla Prediction", "Attention Prediction"
        ])
        wandb.log({"corrected_examples": table})
    
    # Compare performance by word length
    vanilla_df['source_len'] = vanilla_df['source'].apply(len)
    attention_df['source_len'] = attention_df['source'].apply(len)
    
    vanilla_by_len = vanilla_df.groupby('source_len')['correct'].mean().reset_index()
    attention_by_len = attention_df.groupby('source_len')['correct'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(vanilla_by_len['source_len'], vanilla_by_len['correct'], 'b-', label='Vanilla')
    plt.plot(attention_by_len['source_len'], attention_by_len['correct'], 'r-', label='Attention')
    plt.xlabel('Source Word Length')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Word Length: Vanilla vs Attention')
    plt.legend()
    plt.savefig('vanilla_vs_attention_by_length.png')
    wandb.log({"vanilla_vs_attention_by_length": wandb.Image(plt)})
    
    return vanilla_accuracy, attention_accuracy, corrected_examples

def visualize_attention(model, test_loader, data_dict, device, run_id):
    model.eval()
    
    # Create directory for attention visualizations
    os.makedirs('predictions/attention/visualizations', exist_ok=True)
    
    # Get a batch of data
    for batch in test_loader:
        src = batch['source'].to(device)
        trg = batch['target'].to(device)
        src_texts = batch['source_text']
        trg_texts = batch['target_text']
        
        batch_size = src.shape[0]
        max_examples = min(9, batch_size)  # Visualize up to 9 examples
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flat
        
        for i in range(max_examples):
            # Get single example
            src_i = src[i].unsqueeze(0)
            trg_i = trg[i].unsqueeze(0)
            
            # Get encoder outputs and initial hidden state
            encoder_outputs, hidden, cell = model.encoder(src_i)
            if model.cell_type != 'lstm':
                cell = None
            
            # Initialize with < SOS > token
            decoder_input = trg_i[:, 0]
            
            # For storing attention weights
            attentions = []
            decoded_chars = []
            
            # Decode one character at a time
            for t in range(1, trg_i.shape[1]):
                if model.cell_type == 'lstm':
                    decoder_output, hidden, cell, attn_weights = model.decoder(
                        decoder_input, hidden, encoder_outputs, cell
                    )
                else:
                    decoder_output, hidden, _, attn_weights = model.decoder(
                        decoder_input, hidden, encoder_outputs
                    )
                
                top1 = decoder_output.argmax(1)
                decoder_input = top1
                
                # Store attention weights
                attentions.append(attn_weights.squeeze().cpu().detach().numpy())
                
                # Get the predicted character
                char_idx = top1.item()
                if char_idx == data_dict['target_vocab']['<EOS>']:
                    break
                if char_idx != data_dict['target_vocab']['<PAD>'] and char_idx != data_dict['target_vocab']['<UNK>']:
                    decoded_chars.append(data_dict['inv_target_vocab'][char_idx])
            
            # Create attention heatmap
            if len(attentions) > 0:
                attention_matrix = np.array(attentions)
                
                # Get source and target characters for labeling
                src_chars = list(src_texts[i])
                tgt_chars = decoded_chars
                
                # Plot heatmap
                sns.heatmap(
                    attention_matrix, 
                    ax=axes[i],
                    cmap='viridis',
                    xticklabels=src_chars,
                    yticklabels=tgt_chars
                )
                
                axes[i].set_title(f'Source: {src_texts[i]}, Target: {trg_texts[i]}')
                axes[i].set_xlabel('Source')
                axes[i].set_ylabel('Output')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f'predictions/attention/visualizations/attention_heatmap-{run_id}.png')
        wandb.log({"attention_heatmap": wandb.Image(plt)})
        
        # Only process one batch
        break

def visualize_connectivity(model, test_loader, data_dict, device, run_id):
    """
    Visualize the connectivity between encoder and decoder
    Inspired by the "Connectivity" visualization in the assignment
    """
    model.eval()
    
    # Create directory for connectivity visualizations
    os.makedirs('predictions/attention/connectivity', exist_ok=True)
    
    # Get a batch of data
    for batch in test_loader:
        src = batch['source'].to(device)
        trg = batch['target'].to(device)
        src_texts = batch['source_text']
        trg_texts = batch['target_text']
        
        # Select a single example
        idx = 0
        src_i = src[idx].unsqueeze(0)
        trg_i = trg[idx].unsqueeze(0)
        src_text = src_texts[idx]
        trg_text = trg_texts[idx]
        
        # Get encoder outputs and initial hidden state
        encoder_outputs, hidden, cell = model.encoder(src_i)
        if model.cell_type != 'lstm':
            cell = None
        
        # Initialize with < SOS > token
        decoder_input = trg_i[:, 0]
        
        # For storing attention weights
        attentions = []
        decoded_chars = []
        
        # Decode one character at a time
        for t in range(1, trg_i.shape[1]):
            if model.cell_type == 'lstm':
                decoder_output, hidden, cell, attn_weights = model.decoder(
                    decoder_input, hidden, encoder_outputs, cell
                )
            else:
                decoder_output, hidden, _, attn_weights = model.decoder(
                    decoder_input, hidden, encoder_outputs
                )
            
            top1 = decoder_output.argmax(1)
            decoder_input = top1
            
            # Store attention weights
            attentions.append(attn_weights.squeeze().cpu().detach().numpy())
            
            # Get the predicted character
            char_idx = top1.item()
            if char_idx == data_dict['target_vocab']['<EOS>']:
                break
            if char_idx != data_dict['target_vocab']['<PAD>'] and char_idx != data_dict['target_vocab']['<UNK>']:
                decoded_chars.append(data_dict['inv_target_vocab'][char_idx])
        
        # Create connectivity visualization
        if len(attentions) > 0:
            attention_matrix = np.array(attentions)
            
            # Get source and target characters for labeling
            src_chars = list(src_text)
            tgt_chars = decoded_chars
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot connectivity lines
            for i, tgt_char in enumerate(tgt_chars):
                # Get attention weights for this output character
                weights = attention_matrix[i]
                
                # Find the source character with highest attention
                max_idx = np.argmax(weights)
                
                # Plot a line from source to target
                ax.plot([max_idx, i], [0, 1], 'r-', alpha=0.5, linewidth=2)
                
                # Add more lines for other significant attention weights
                for j, w in enumerate(weights):
                    if j != max_idx and w > 0.2:  # Threshold for significant attention
                        ax.plot([j, i], [0, 1], 'b-', alpha=0.3 * w, linewidth=1)
            
            # Plot source characters
            for i, char in enumerate(src_chars):
                ax.text(i, 0, char, ha='center', va='center', fontsize=14)
            
            # Plot target characters
            for i, char in enumerate(tgt_chars):
                ax.text(i, 1, char, ha='center', va='center', fontsize=14)
            
            # Set axis limits
            ax.set_xlim(-1, max(len(src_chars), len(tgt_chars)))
            ax.set_ylim(-0.2, 1.2)
            
            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add labels
            ax.text(-0.05, 0, 'Source', ha='right', va='center', fontsize=12)
            ax.text(-0.05, 1, 'Target', ha='right', va='center', fontsize=12)
            
            # Add title
            plt.title(f'Connectivity Visualization: {src_text} → {"".join(decoded_chars)}')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(f'predictions/attention/connectivity/connectivity-{run_id}.png')
            wandb.log({"connectivity_visualization": wandb.Image(plt)})
        
        # Only process one example
        break
