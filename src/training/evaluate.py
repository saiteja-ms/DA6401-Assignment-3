import torch
import torch.nn as nn
from tqdm import tqdm
from ..models import AttentionSeq2Seq

def evaluate(model: nn.Module, iterator: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """
    Evaluates the model on a given dataset iterator.
    Calculates average loss over the dataset, using attestation counts as weights.
    """
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating Loss"):
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            attestation = batch['attestation'].to(device)  # Get attestation counts

            # Pass source and target to the model
            if isinstance(model, AttentionSeq2Seq):
                output, _ = model(src, trg, 0)
            else:
                output = model(src, trg, 0)

            output_dim = output.shape[-1]

            # Slice the output and target to remove the first timestep (SOS)
            output_seq = output[:, 1:, :]
            trg_seq = trg[:, 1:]

            # Check for empty sequences after slicing
            if trg_seq.numel() == 0:
                continue

            # Calculate unweighted loss
            loss = criterion(output_seq.reshape(-1, output_dim), trg_seq.reshape(-1))
            
            # Apply attestation weights to the loss
            seq_len = trg_seq.shape[1]
            attestation_weights = attestation.repeat_interleave(seq_len)
            
            # Normalize weights to sum to batch size
            attestation_weights = attestation_weights * (attestation_weights.size(0) / attestation_weights.sum())
            
            # Apply weights to loss
            weighted_loss = (loss * attestation_weights).mean()
            
            epoch_loss += weighted_loss.item()

    return epoch_loss / len(iterator) if len(iterator) > 0 else 0.0

def calculate_accuracy(model: nn.Module, iterator: torch.utils.data.DataLoader, inv_target_vocab: dict, device: torch.device) -> tuple[float, list[dict]]:
    """
    Calculates word-level accuracy (exact match) and generates predictions.
    Performs greedy decoding step-by-step.
    Includes attestation counts in the predictions.
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []

    # Get special token indices using .get() with fallbacks
    target_vocab = {v: k for k, v in inv_target_vocab.items()}
    
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

    # Determine maximum prediction length
    max_prediction_length = getattr(iterator.dataset, 'max_len', 50)
    if max_prediction_length < 10:
        max_prediction_length = 50

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Calculating Accuracy"):
            src = batch['source'].to(device)
            trg_texts = batch['target_text']
            attestation_counts = batch['attestation']  # Get attestation counts

            # Get the actual batch size for the current batch
            current_batch_size = src.shape[0]

            # Handle empty batches gracefully
            if current_batch_size == 0:
                continue

            # --- Encoder Step ---
            if hasattr(model, 'cell_type') and model.cell_type == 'lstm':
                encoder_outputs, hidden, cell = model.encoder(src)
            else:
                encoder_outputs, hidden, _ = model.encoder(src)
                cell = None

            # --- Decoder Step-by-Step Decoding (Greedy Search) ---
            decoder_input = torch.full((current_batch_size,), sos_idx, dtype=torch.long, device=device)

            batch_decoded_indices = [[] for _ in range(current_batch_size)]
            finished_decoding = [False] * current_batch_size
            
            # For storing attention weights if using attention model
            batch_attention_weights = [[] for _ in range(current_batch_size)] if isinstance(model, AttentionSeq2Seq) else None

            for t in range(max_prediction_length):
                if isinstance(model, AttentionSeq2Seq):
                    if hasattr(model, 'cell_type') and model.cell_type == 'lstm':
                        decoder_output, hidden, cell, attn_weights = model.decoder(
                            decoder_input, hidden, encoder_outputs, cell
                        )
                    else:
                        decoder_output, hidden, _, attn_weights = model.decoder(
                            decoder_input, hidden, encoder_outputs
                        )
                else:
                    if hasattr(model, 'cell_type') and model.cell_type == 'lstm':
                        decoder_output, hidden, cell = model.decoder(
                            decoder_input, hidden, cell
                        )
                    else:
                        decoder_output, hidden, _ = model.decoder(
                            decoder_input, hidden
                        )

                # Get the predicted token index for this step
                top1 = decoder_output.argmax(1)
                
                # Update decoded indices and finished status for each sequence in the batch
                for i in range(current_batch_size):
                    if not finished_decoding[i]:
                        predicted_token_idx = top1[i].item()
                        batch_decoded_indices[i].append(predicted_token_idx)
                        
                        # Store attention weights if using attention model
                        if isinstance(model, AttentionSeq2Seq):
                            batch_attention_weights[i].append(attn_weights[i].cpu().numpy())

                        # Check for EOS
                        if predicted_token_idx == eos_idx:
                            finished_decoding[i] = True

                # The input for the *next* step is the tokens predicted in this step
                decoder_input = top1

                # Stop early if all sequences have finished decoding
                if all(finished_decoding):
                    break

            # --- Post-process decoded indices to get predicted strings ---
            for i in range(current_batch_size):
                pred_chars = []
                for idx in batch_decoded_indices[i]:
                    if idx == eos_idx:
                        break
                    if idx != pad_idx and idx != unk_idx:
                        pred_char = inv_target_vocab.get(idx, None)
                        if pred_char is not None and pred_char not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '< SOS >']:
                            pred_chars.append(pred_char)

                pred_text = ''.join(pred_chars)
                original_target_text = trg_texts[i]
                
                # Word-level accuracy (exact match)
                is_correct = (pred_text == original_target_text)
                if is_correct:
                    correct += 1
                total += 1

                # Store prediction details with attestation count
                prediction_info = {
                    'source': batch['source_text'][i],
                    'target': batch['target_text'][i],
                    'prediction': pred_text,
                    'correct': is_correct,
                    'attestation': attestation_counts[i].item()  # Include attestation count
                }
                
                # Add attention weights if available
                if isinstance(model, AttentionSeq2Seq):
                    prediction_info['attention_weights'] = batch_attention_weights[i]
                
                predictions.append(prediction_info)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, predictions
