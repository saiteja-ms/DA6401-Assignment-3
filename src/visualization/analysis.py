import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
import torch
import os
import io

def analyze_errors(predictions_file: str):
    """
    Analyzes errors from a prediction JSON file.
    Logs accuracy by length, attestation count, and character confusion matrix to wandb.
    """
    if not os.path.exists(predictions_file):
        print(f"Error: Predictions file not found at {predictions_file}")
        return 0.0, []

    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(predictions)

    if df.empty:
        print(f"Warning: No predictions found in {predictions_file}")
        return 0.0, []
        
    # Calculate overall accuracy
    accuracy = df['correct'].mean()
    print(f"Overall accuracy: {accuracy:.4f}")

    # Analyze errors by word length
    df['source_len'] = df['source'].apply(len)
    df['target_len'] = df['target'].apply(len)
    df['prediction_len'] = df['prediction'].apply(len)

    # Group by source length
    if not df.empty and 'source_len' in df.columns and 'correct' in df.columns:
        length_accuracy = df.groupby('source_len')['correct'].mean().reset_index()

        plt.figure(figsize=(12, 6))
        plt.bar(length_accuracy['source_len'], length_accuracy['correct'])
        plt.xlabel('Source Word Length')
        plt.ylabel('Exact Match Accuracy')
        plt.title('Accuracy by Source Word Length')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig('accuracy_by_length.png')
        if wandb.run:
            wandb.log({"accuracy_by_length": wandb.Image(plt)})
        plt.close()
    else:
        print("DataFrame is empty or missing required columns for length analysis.")

    # Analyze accuracy by attestation count if available
    if 'attestation' in df.columns:
        attestation_accuracy = df.groupby('attestation')['correct'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(attestation_accuracy['attestation'], attestation_accuracy['correct'])
        plt.xlabel('Attestation Count')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Attestation Count')
        plt.savefig('accuracy_by_attestation.png')
        if wandb.run:
            wandb.log({"accuracy_by_attestation": wandb.Image(plt)})
        plt.close()
        
        # Print correlation between attestation and accuracy
        correlation = df['attestation'].corr(df['correct'])
        print(f"Correlation between attestation count and accuracy: {correlation:.4f}")
        if wandb.run:
            wandb.log({"attestation_accuracy_correlation": correlation})

    # Analyze character-level errors
    error_pairs = []
    for _, row in df[~df['correct']].iterrows():
        target = row['target']
        pred = row['prediction']

        min_len = min(len(target), len(pred))
        max_len = max(len(target), len(pred))
        
        for i in range(max_len):
            t_char = target[i] if i < len(target) else '<MISSING>'
            p_char = pred[i] if i < len(pred) else '<EXTRA>'

            if t_char != p_char:
                error_pairs.append((t_char, p_char))

    # Count error pairs
    error_counts = {}
    for pair in error_pairs:
        if pair in error_counts:
            error_counts[pair] += 1
        else:
            error_counts[pair] = 1
    
    most_common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    print("\nMost common character-level errors (Actual -> Predicted):")
    if most_common_errors:
        for (target, pred), count in most_common_errors:
            print(f"  '{target}' -> '{pred}': {count} times")
    else:
        print("No character-level errors found.")

    # Create confusion matrix for most common characters
    all_chars = set([char for pair in error_pairs for char in pair])
    labels = sorted(list(all_chars))

    if labels:
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_idx = {label: i for i, label in enumerate(labels)}

        for (t, p), count in error_counts.items():
            if t in label_to_idx and p in label_to_idx:
                cm[label_to_idx[t], label_to_idx[p]] += count

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel('Predicted Character')
        plt.ylabel('Actual Character')
        plt.title('Character Confusion Matrix (Errors Only)')
        plt.tight_layout()
        plt.savefig('character_confusion_matrix.png')
        if wandb.run:
            wandb.log({"character_confusion_matrix": wandb.Image(plt)})
        plt.close()
    else:
        print("Not enough data to build a character confusion matrix.")

    # Create a sample grid/table of predictions
    num_samples = 10
    correct_df = df[df['correct']]
    incorrect_df = df[~df['correct']]
    
    correct_samples = correct_df.sample(min(num_samples // 2, len(correct_df)), random_state=42) if not correct_df.empty else pd.DataFrame()
    incorrect_samples = incorrect_df.sample(min(num_samples - len(correct_samples), len(incorrect_df)), random_state=42) if not incorrect_df.empty else pd.DataFrame()

    samples_df = pd.concat([correct_samples, incorrect_samples]).sample(frac=1, random_state=42).reset_index(drop=True) if not pd.concat([correct_samples, incorrect_samples]).empty else pd.DataFrame()

    if not samples_df.empty:
        table_data = []
        for _, sample in samples_df.iterrows():
            table_data.append([
                sample['source'],
                sample['target'],
                sample['prediction'],
                '✓' if sample['correct'] else '✗',
                sample.get('attestation', 'N/A')  # Include attestation count if available
            ])

        table = wandb.Table(data=table_data, columns=["Source", "Target", "Prediction", "Correct", "Attestation"])
        if wandb.run:
            wandb.log({"prediction_samples": table})
    else:
        print("No samples available for prediction table.")

    return accuracy, most_common_errors

def compare_models(vanilla_predictions_file: str, attention_predictions_file: str):
    """
    Compares prediction files from vanilla and attention models.
    Logs comparison plots and tables to wandb.
    """
    vanilla_exists = os.path.exists(vanilla_predictions_file)
    attention_exists = os.path.exists(attention_predictions_file)

    if not vanilla_exists and not attention_exists:
        print(f"Error: Neither prediction file found: {vanilla_predictions_file}, {attention_predictions_file}")
        return 0.0, 0.0, []

    vanilla_df = pd.DataFrame()
    attention_df = pd.DataFrame()

    if vanilla_exists:
        with open(vanilla_predictions_file, 'r', encoding='utf-8') as f:
            vanilla_predictions = json.load(f)
            vanilla_df = pd.DataFrame(vanilla_predictions)
            vanilla_df = vanilla_df.sort_values('source').reset_index(drop=True)
            print(f"Loaded {len(vanilla_df)} predictions from vanilla model.")

    if attention_exists:
        with open(attention_predictions_file, 'r', encoding='utf-8') as f:
            attention_predictions = json.load(f)
            attention_df = pd.DataFrame(attention_predictions)
            attention_df = attention_df.sort_values('source').reset_index(drop=True)
            print(f"Loaded {len(attention_df)} predictions from attention model.")

    if vanilla_df.empty and attention_df.empty:
        print("No prediction data loaded for comparison.")
        return 0.0, 0.0, []

    # Calculate accuracies
    vanilla_accuracy = vanilla_df['correct'].mean() if not vanilla_df.empty else 0.0
    attention_accuracy = attention_df['correct'].mean() if not attention_df.empty else 0.0
    
    print(f"\nVanilla model accuracy: {vanilla_accuracy:.4f}")
    print(f"Attention model accuracy: {attention_accuracy:.4f}")

    corrected_examples = []
    worsened_examples = []

    # Find examples where attention model corrected vanilla model's errors
    if not vanilla_df.empty and not attention_df.empty and len(vanilla_df) == len(attention_df):
        comparison_df = vanilla_df.rename(columns={'correct': 'vanilla_correct', 'prediction': 'vanilla_prediction'}).copy()
        comparison_df['attention_correct'] = attention_df['correct']
        comparison_df['attention_prediction'] = attention_df['prediction']
        comparison_df['target'] = attention_df['target']
        
        # Add attestation counts if available
        if 'attestation' in vanilla_df.columns:
            comparison_df['attestation'] = vanilla_df['attestation']

        # Examples corrected by attention: vanilla was wrong, attention is correct
        corrected_examples = comparison_df[
            (comparison_df['vanilla_correct'] == False) & (comparison_df['attention_correct'] == True)
        ].to_dict('records')

        # Examples worsened by attention: vanilla was correct, attention is wrong
        worsened_examples = comparison_df[
            (comparison_df['vanilla_correct'] == True) & (comparison_df['attention_correct'] == False)
        ].to_dict('records')

        print(f"\nNumber of examples corrected by attention model: {len(corrected_examples)}")
        print(f"Number of examples worsened by attention model: {len(worsened_examples)}")

        # Create a table of corrected examples
        if corrected_examples:
            table_data = []
            for example in corrected_examples[:15]:
                row = [
                    example['source'],
                    example['target'],
                    example['vanilla_prediction'],
                    example['attention_prediction']
                ]
                # Add attestation count if available
                if 'attestation' in example:
                    row.append(example['attestation'])
                table_data.append(row)

            columns = ["Source", "Target", "Vanilla Prediction", "Attention Prediction"]
            if 'attestation' in corrected_examples[0]:
                columns.append("Attestation")
                
            table = wandb.Table(data=table_data, columns=columns)
            if wandb.run:
                wandb.log({"corrected_examples_by_attention": table})
        else:
            print("No examples corrected by attention.")

        # Create a table of worsened examples
        if worsened_examples:
            table_data = []
            for example in worsened_examples[:15]:
                row = [
                    example['source'],
                    example['target'],
                    example['vanilla_prediction'],
                    example['attention_prediction']
                ]
                # Add attestation count if available
                if 'attestation' in example:
                    row.append(example['attestation'])
                table_data.append(row)

            columns = ["Source", "Target", "Vanilla Prediction", "Attention Prediction"]
            if 'attestation' in worsened_examples[0]:
                columns.append("Attestation")
                
            table = wandb.Table(data=table_data, columns=columns)
            if wandb.run:
                wandb.log({"worsened_examples_by_attention": table})
        else:
            print("No examples worsened by attention.")

        # Compare performance by word length
        if not vanilla_df.empty and 'source' in vanilla_df.columns:
            vanilla_df['source_len'] = vanilla_df['source'].apply(len)
            vanilla_by_len = vanilla_df.groupby('source_len')['correct'].mean().reset_index().rename(columns={'correct': 'Vanilla Accuracy'})
        else: 
            vanilla_by_len = pd.DataFrame(columns=['source_len', 'Vanilla Accuracy'])

        if not attention_df.empty and 'source' in attention_df.columns:
            attention_df['source_len'] = attention_df['source'].apply(len)
            attention_by_len = attention_df.groupby('source_len')['correct'].mean().reset_index().rename(columns={'correct': 'Attention Accuracy'})
        else: 
            attention_by_len = pd.DataFrame(columns=['source_len', 'Attention Accuracy'])

        # Merge for plotting
        compare_by_len = pd.merge(vanilla_by_len, attention_by_len, on='source_len', how='outer').sort_values('source_len')

        if not compare_by_len.empty:
            plt.figure(figsize=(14, 7))
            plt.plot(compare_by_len['source_len'], compare_by_len['Vanilla Accuracy'], 'o-', label='Vanilla')
            plt.plot(compare_by_len['source_len'], compare_by_len['Attention Accuracy'], 'o-', label='Attention')
            plt.xlabel('Source Word Length')
            plt.ylabel('Exact Match Accuracy')
            plt.title('Accuracy by Word Length: Vanilla vs Attention')
            plt.legend()
            plt.grid(True, linestyle='--')
            plt.tight_layout()
            plt.savefig('vanilla_vs_attention_by_length.png')
            if wandb.run:
                wandb.log({"vanilla_vs_attention_by_length": wandb.Image(plt)})
            plt.close()
        else:
            print("Not enough data to plot accuracy by length comparison.")
            
        # Compare performance by attestation count if available
        if 'attestation' in vanilla_df.columns and 'attestation' in attention_df.columns:
            vanilla_by_att = vanilla_df.groupby('attestation')['correct'].mean().reset_index().rename(columns={'correct': 'Vanilla Accuracy'})
            attention_by_att = attention_df.groupby('attestation')['correct'].mean().reset_index().rename(columns={'correct': 'Attention Accuracy'})
            
            compare_by_att = pd.merge(vanilla_by_att, attention_by_att, on='attestation', how='outer').sort_values('attestation')
            
            if not compare_by_att.empty:
                plt.figure(figsize=(14, 7))
                plt.plot(compare_by_att['attestation'], compare_by_att['Vanilla Accuracy'], 'o-', label='Vanilla')
                plt.plot(compare_by_att['attestation'], compare_by_att['Attention Accuracy'], 'o-', label='Attention')
                plt.xlabel('Attestation Count')
                plt.ylabel('Exact Match Accuracy')
                plt.title('Accuracy by Attestation Count: Vanilla vs Attention')
                plt.legend()
                plt.grid(True, linestyle='--')
                plt.tight_layout()
                plt.savefig('vanilla_vs_attention_by_attestation.png')
                if wandb.run:
                    wandb.log({"vanilla_vs_attention_by_attestation": wandb.Image(plt)})
                plt.close()
            else:
                print("Not enough data to plot accuracy by attestation comparison.")

    else:
        print("Cannot compare models: DataFrames are empty or have different lengths.")

    return vanilla_accuracy, attention_accuracy, corrected_examples

def visualize_attention(model, test_loader, data_dict, device, run_id):
    """
    Generates and logs attention heatmaps for a few examples from the test set.
    Assumes model is an AttentionSeq2Seq model.
    """
    from ..models import AttentionSeq2Seq
    if not isinstance(model, AttentionSeq2Seq):
        print("Model is not an AttentionSeq2Seq model. Skipping attention visualization.")
        return

    model.eval()

    # Get special token indices
    target_vocab = data_dict['target_vocab']
    inv_target_vocab = data_dict['inv_target_vocab']
    
    # Try different possible token names for special tokens
    sos_idx = None
    for token in ['<SOS>', '< SOS >', 'SOS']:
        if token in target_vocab:
            sos_idx = target_vocab[token]
            break
    if sos_idx is None:
        sos_idx = 2  # Default SOS index
        
    eos_idx = None
    for token in ['<EOS>', '< EOS >', 'EOS']:
        if token in target_vocab:
            eos_idx = target_vocab[token]
            break
    if eos_idx is None:
        eos_idx = 3  # Default EOS index
        
    pad_idx = target_vocab.get('<PAD>', 0)
    unk_idx = target_vocab.get('<UNK>', 1)

    # Create directory for attention visualizations
    vis_dir = 'predictions/attention/visualizations'
    os.makedirs(vis_dir, exist_ok=True)

    # Get a batch of data to visualize
    print("\nGenerating attention heatmaps...")
    with torch.no_grad():
        try:
            batch = next(iter(test_loader))
        except StopIteration:
            print("Test loader is empty, cannot generate visualizations.")
            return

        src = batch['source'].to(device)
        trg_texts = batch['target_text']
        src_texts = batch['source_text']
        attestation_counts = batch['attestation']  # Get attestation counts

        batch_size = src.shape[0]
        max_examples_to_plot = min(9, batch_size)

        if max_examples_to_plot == 0:
            print("No data in the batch for visualization.")
            return

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flat

        # --- Get Encoder Outputs ---
        if model.cell_type == 'lstm':
            encoder_outputs, hidden, cell = model.encoder(src)
        else:
            encoder_outputs, hidden, _ = model.encoder(src)
            cell = None

        # --- Decode step by step for each example to get attention weights ---
        decoder_input = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)

        all_batch_attentions = [[] for _ in range(batch_size)]
        batch_predicted_indices = [[] for _ in range(batch_size)]
        finished_decoding = [False] * batch_size

        # Determine max decoding length
        max_decode_steps = max(getattr(test_loader.dataset, 'max_len', 50), max(len(t) for t in trg_texts) + 5)
        max_decode_steps = min(max_decode_steps, 100)

        # Decode step by step
        for t in range(max_decode_steps):
            if model.cell_type == 'lstm':
                decoder_output, hidden, cell, attn_weights = model.decoder(
                    decoder_input, hidden, encoder_outputs, cell
                )
            else:
                decoder_output, hidden, _, attn_weights = model.decoder(
                    decoder_input, hidden, encoder_outputs
                )

            # Get predicted token for this step
            top1 = decoder_output.argmax(1)

            # Store attention weights and predicted indices for each sequence that is not finished
            for i in range(batch_size):
                if not finished_decoding[i]:
                    predicted_token_idx = top1[i].item()
                    all_batch_attentions[i].append(attn_weights[i].cpu().detach().numpy())
                    batch_predicted_indices[i].append(predicted_token_idx)

                    # Check for EOS
                    if predicted_token_idx == eos_idx:
                        finished_decoding[i] = True

            # The input for the *next* step is the tokens predicted in this step
            decoder_input = top1

            # Stop if all are finished
            if all(finished_decoding):
                break

        # --- Generate Plots for Examples ---
        plotted_count = 0
        for i in range(batch_size):
            if plotted_count >= max_examples_to_plot:
                break

            src_text = src_texts[i]
            original_target_text = trg_texts[i]
            attestation_count = attestation_counts[i].item()  # Get attestation count for this example

            # Get the sequence of predicted characters for plotting, stopping at EOS and ignoring PAD/UNK
            predicted_chars_for_plot = []
            attention_matrix_for_plot = []

            # Iterate through predicted indices and collected attentions for this example
            if len(batch_predicted_indices[i]) != len(all_batch_attentions[i]):
                print(f"Warning: Mismatch between predicted tokens ({len(batch_predicted_indices[i])}) and attentions ({len(all_batch_attentions[i])}) for example {i}. Skipping heatmap.")
                continue

            for idx, attn_vector in zip(batch_predicted_indices[i], all_batch_attentions[i]):
                if idx == eos_idx:
                    break
                if idx != pad_idx and idx != unk_idx:
                    pred_char = inv_target_vocab.get(idx, None)
                    if pred_char is not None and pred_char not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '< SOS >']:
                        predicted_chars_for_plot.append(pred_char)
                        attention_matrix_for_plot.append(attn_vector)

            # Reconstruct the full predicted text string
            full_predicted_chars = []
            for idx in batch_predicted_indices[i]:
                if idx == eos_idx: break
                pred_char = inv_target_vocab.get(idx, None)
                if pred_char is not None and pred_char not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '< SOS >']:
                    full_predicted_chars.append(pred_char)
            full_predicted_text = ''.join(full_predicted_chars)

            # Create attention heatmap matrix if we have data
            if attention_matrix_for_plot:
                attention_matrix_for_plot = np.vstack(attention_matrix_for_plot)

                # Labels for the heatmap
                src_chars = list(src_text)
                tgt_chars_for_plot = predicted_chars_for_plot

                # Plot heatmap on the current axis
                ax = axes[plotted_count]
                sns.heatmap(
                    attention_matrix_for_plot,
                    ax=ax,
                    cmap='viridis',
                    xticklabels=src_chars,
                    yticklabels=tgt_chars_for_plot,
                    cbar=False
                )

                # Include attestation count in the title
                ax.set_title(f'Src: {src_text}\nPred: {full_predicted_text}\nAttestation: {attestation_count}', fontsize=10)
                ax.set_xlabel('Source Characters', fontsize=8)
                ax.set_ylabel('Predicted Characters', fontsize=8)
                ax.tick_params(axis='x', rotation=90, labelsize=6)
                ax.tick_params(axis='y', rotation=0, labelsize=6)

                plotted_count += 1

            else:
                print(f"No valid characters predicted for example {i} for plotting heatmap.")

        # Hide any unused subplots
        for j in range(plotted_count, 9):
            fig.delaxes(axes[j])

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        if wandb.run:
            wandb.log({"attention_heatmaps": wandb.Image(buf)})
        print(f"Logged {plotted_count} attention heatmaps.")

def visualize_connectivity(model, test_loader, data_dict, device, run_id=None, save_dir='predictions/attention/connectivity'):
    """
    Visualize the connectivity between encoder and decoder outputs using attention weights.
    Similar to the visualization in the Distill article.
    """
    from ..models import AttentionSeq2Seq
    if not isinstance(model, AttentionSeq2Seq):
        print("Model is not an AttentionSeq2Seq model. Skipping connectivity visualization.")
        return

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Get special token indices
    target_vocab = data_dict['target_vocab']
    inv_target_vocab = data_dict['inv_target_vocab']
    
    # Try different possible token names for special tokens
    sos_idx = None
    for token in ['< SOS >', '< SOS >', 'SOS']:
        if token in target_vocab:
            sos_idx = target_vocab[token]
            break
    if sos_idx is None:
        sos_idx = 2  # Default SOS index
        
    eos_idx = None
    for token in ['<EOS>', '< EOS >', 'EOS']:
        if token in target_vocab:
            eos_idx = target_vocab[token]
            break
    if eos_idx is None:
        eos_idx = 3  # Default EOS index
        
    pad_idx = target_vocab.get('<PAD>', 0)
    unk_idx = target_vocab.get('<UNK>', 1)

    with torch.no_grad():
        # Get a batch of examples
        batch_idx = 0
        for batch in test_loader:
            if batch_idx >= 5:  # Limit to 5 examples
                break
                
            src = batch['source'].to(device)
            trg_texts = batch['target_text']
            src_texts = batch['source_text']
            
            # Process one example at a time for clearer visualization
            for i in range(min(3, len(src_texts))):  # Process up to 3 examples per batch
                single_src = src[i:i+1]
                src_text = src_texts[i]
                trg_text = trg_texts[i]
                
                # Get encoder outputs
                if model.cell_type == 'lstm':
                    encoder_outputs, hidden, cell = model.encoder(single_src)
                else:
                    encoder_outputs, hidden, _ = model.encoder(single_src)
                    cell = None
                
                # Initialize with SOS token
                decoder_input = torch.tensor([sos_idx], dtype=torch.long, device=device)
                
                # Store attention weights and decoded tokens
                attention_weights = []
                decoded_tokens = []
                
                # Decode step by step
                max_len = 50
                for _ in range(max_len):
                    if model.cell_type == 'lstm':
                        decoder_output, hidden, cell, attn_weights = model.decoder(
                            decoder_input, hidden, encoder_outputs, cell
                        )
                    else:
                        decoder_output, hidden, _, attn_weights = model.decoder(
                            decoder_input, hidden, encoder_outputs
                        )
                    
                    # Get predicted token
                    top1 = decoder_output.argmax(1)
                    token_idx = top1.item()
                    
                    # Store attention weights
                    attention_weights.append(attn_weights.cpu().numpy()[0])
                    
                    # Store decoded token
                    if token_idx == eos_idx:
                        break
                    
                    if token_idx != pad_idx and token_idx != unk_idx:
                        token = inv_target_vocab.get(token_idx, '')
                        if token not in ['<PAD>', '<UNK>', '< SOS >', '<EOS>']:
                            decoded_tokens.append(token)
                    
                    # Next input is current prediction
                    decoder_input = top1
                
                # Create connectivity visualization
                if len(attention_weights) > 0 and len(decoded_tokens) > 0:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot source characters at the bottom
                    src_chars = list(src_text)
                    for j, char in enumerate(src_chars):
                        plt.text(j, 0, char, ha='center', va='center', fontsize=14)
                    
                    # Plot target characters at the top
                    for j, char in enumerate(decoded_tokens):
                        plt.text(j, 1, char, ha='center', va='center', fontsize=14)
                    
                    # Draw connections based on attention weights
                    attention_matrix = np.array(attention_weights)
                    
                    # For each target character
                    for j, target_char in enumerate(decoded_tokens):
                        if j >= len(attention_weights):
                            continue
                            
                        # Get attention weights for this character
                        weights = attention_weights[j]
                        
                        # Find the source character with highest attention
                        max_idx = np.argmax(weights)
                        
                        # Draw a strong line for the highest attention
                        plt.plot([max_idx, j], [0, 1], 'r-', alpha=0.7, linewidth=2)
                        
                        # Draw fainter lines for other significant attentions
                        for k, weight in enumerate(weights):
                            if k != max_idx and weight > 0.1:  # Threshold for significant attention
                                plt.plot([k, j], [0, 1], 'b-', alpha=0.3 * weight, linewidth=1)
                    
                    # Set axis limits
                    plt.xlim(-1, max(len(src_chars), len(decoded_tokens)))
                    plt.ylim(-0.2, 1.2)
                    
                    # Remove axis ticks
                    plt.xticks([])
                    plt.yticks([])
                    
                    # Add labels
                    plt.text(-0.05, 0, 'Source', ha='right', va='center', fontsize=12)
                    plt.text(-0.05, 1, 'Target', ha='right', va='center', fontsize=12)
                    
                    # Add title
                    plt.title(f'Connectivity Visualization\nSource: {src_text} → Prediction: {"".join(decoded_tokens)}')
                    
                    # Save figure
                    filename = f'connectivity_{batch_idx}_{i}.png'
                    if run_id:
                        filename = f'connectivity_{run_id}_{batch_idx}_{i}.png'
                    
                    save_path = os.path.join(save_dir, filename)
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=300)
                    
                    # Log to wandb
                    if wandb.run:
                        wandb.log({f"connectivity_{batch_idx}_{i}": wandb.Image(plt)})
                    
                    plt.close()
                    
                    print(f"Saved connectivity visualization to {save_path}")
            
            batch_idx += 1
