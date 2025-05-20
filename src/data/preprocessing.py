import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TransliterationDataset(Dataset):
    """Custom Dataset for Transliteration pairs with attestation counts."""
    def __init__(self, source_texts, target_texts, attestation_counts, source_vocab, target_vocab, max_len=50):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.attestation_counts = attestation_counts  # Added attestation counts
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len

        # Get special token indices, assuming they are always present after load_dakshina_data
        self.pad_idx = self.target_vocab.get('<PAD>', 0)
        self.unk_idx = self.target_vocab.get('<UNK>', 1)
        self.sos_idx = self.target_vocab.get('<SOS>', 2)
        self.eos_idx = self.target_vocab.get('<EOS>', 3)

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        attestation_count = self.attestation_counts[idx]

        # Convert non-string values to strings safely
        source_text = str(source_text) if not isinstance(source_text, str) else source_text
        target_text = str(target_text) if not isinstance(target_text, str) else target_text

        # Convert characters to indices using .get with UNK fallback
        source_indices = [self.source_vocab.get(char, self.unk_idx) for char in source_text]
        target_indices = [self.target_vocab.get(char, self.unk_idx) for char in target_text]

        # Add SOS and EOS tokens to target sequence
        target_indices = [self.sos_idx] + target_indices + [self.eos_idx]

        # Truncate sequences if longer than max_len
        source_indices = source_indices[:self.max_len]
        target_indices = target_indices[:self.max_len]

        # Pad sequences to max_len using the PAD index
        source_indices += [self.pad_idx] * (self.max_len - len(source_indices))
        target_indices += [self.pad_idx] * (self.max_len - len(target_indices))

        return {
            'source': torch.tensor(source_indices, dtype=torch.long),
            'target': torch.tensor(target_indices, dtype=torch.long),
            'source_text': source_text,
            'target_text': target_text,
            'attestation': torch.tensor(attestation_count, dtype=torch.float)  # Added attestation count as tensor
        }

def load_dakshina_data(language='ta', base_dir='.', max_len=50):
    """
    Load data from the Dakshina dataset for a specific language.
    language: language code (e.g., 'ta' for Tamil)
    base_dir: base directory containing the dataset structure
    max_len: maximum sequence length for padding/truncation
    """
    # Define file paths based on Dakshina dataset structure
    train_file = os.path.join(base_dir, language, "lexicons", f"{language}.translit.sampled.train.tsv")
    dev_file = os.path.join(base_dir, language, "lexicons", f"{language}.translit.sampled.dev.tsv")
    test_file = os.path.join(base_dir, language, "lexicons", f"{language}.translit.sampled.test.tsv")
    
    print(f"Looking for training file at: {train_file}")
    
    # Check if files exist
    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in {os.path.dirname(train_file)} (if directory exists):")
        try:
            if os.path.exists(os.path.dirname(train_file)):
                print(os.listdir(os.path.dirname(train_file)))
            else:
                print(f"Directory {os.path.dirname(train_file)} does not exist")
        except Exception as e:
            print(f"Error listing directory: {e}")
        
        # Create a test dataset if the real one can't be found
        return create_test_dataset(max_len=max_len)
    
    # Load data with error handling
    try:
        # In Dakshina, the format is: native_script \t romanization \t attestation_count
        # We want source (romanization) -> target (native_script)
        train_df = pd.read_csv(train_file, sep='\t', header=None, 
                              names=['target', 'source', 'attestation'], 
                              keep_default_na=False, on_bad_lines='skip')
        dev_df = pd.read_csv(dev_file, sep='\t', header=None, 
                            names=['target', 'source', 'attestation'], 
                            keep_default_na=False, on_bad_lines='skip')
        test_df = pd.read_csv(test_file, sep='\t', header=None, 
                             names=['target', 'source', 'attestation'], 
                             keep_default_na=False, on_bad_lines='skip')

        # Convert attestation counts to integers
        train_df['attestation'] = train_df['attestation'].astype(int)
        dev_df['attestation'] = dev_df['attestation'].astype(int)
        test_df['attestation'] = test_df['attestation'].astype(int)

        # Convert any non-string values to strings explicitly
        train_df['source'] = train_df['source'].apply(str)
        train_df['target'] = train_df['target'].apply(str)
        dev_df['source'] = dev_df['source'].apply(str)
        dev_df['target'] = dev_df['target'].apply(str)
        test_df['source'] = test_df['source'].apply(str)
        test_df['target'] = test_df['target'].apply(str)

        # Build vocabularies from the training data
        source_chars = set()
        target_chars = set()

        for text in train_df['source']:
            if isinstance(text, str):
                source_chars.update(text)

        for text in train_df['target']:
            if isinstance(text, str):
                target_chars.update(text)

        # Create vocabulary dictionaries with consistent special tokens
        source_vocab = {}
        target_vocab = {}

        # Add special tokens first with known indices
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        for i, token in enumerate(special_tokens):
            source_vocab[token] = i
            target_vocab[token] = i

        # Add sorted unique characters from data
        for char in sorted(list(source_chars)):
            if char not in source_vocab:
                source_vocab[char] = len(source_vocab)

        for char in sorted(list(target_chars)):
            if char not in target_vocab:
                target_vocab[char] = len(target_vocab)

        # Print debug information about the vocabularies
        print("Special tokens in vocabulary:")
        print(f"Source vocab keys: {list(source_vocab.keys())[:10]}")
        print(f"Target vocab keys: {list(target_vocab.keys())[:10]}")

        # Create inverse vocabularies for decoding
        inv_source_vocab = {v: k for k, v in source_vocab.items()}
        inv_target_vocab = {v: k for k, v in target_vocab.items()}

        # Create datasets using the loaded data and created vocabs
        # Include attestation counts in the dataset creation
        train_dataset = TransliterationDataset(
            train_df['source'].tolist(),
            train_df['target'].tolist(),
            train_df['attestation'].tolist(),  # Pass attestation counts
            source_vocab,
            target_vocab,
            max_len=max_len
        )

        dev_dataset = TransliterationDataset(
            dev_df['source'].tolist(),
            dev_df['target'].tolist(),
            dev_df['attestation'].tolist(),  # Pass attestation counts
            source_vocab,
            target_vocab,
            max_len=max_len
        )

        test_dataset = TransliterationDataset(
            test_df['source'].tolist(),
            test_df['target'].tolist(),
            test_df['attestation'].tolist(),  # Pass attestation counts
            source_vocab,
            target_vocab,
            max_len=max_len
        )

        print(f"Successfully loaded Dakshina dataset for {language}")
        print(f"Train set: {len(train_dataset)} examples")
        print(f"Dev set: {len(dev_dataset)} examples")
        print(f"Test set: {len(test_dataset)} examples")
        print(f"Source vocabulary size: {len(source_vocab)}")
        print(f"Target vocabulary size: {len(target_vocab)}")
        print(f"Max sequence length: {max_len}")

        return {
            'train_dataset': train_dataset,
            'dev_dataset': dev_dataset,
            'test_dataset': test_dataset,
            'source_vocab': source_vocab,
            'target_vocab': target_vocab,
            'inv_source_vocab': inv_source_vocab,
            'inv_target_vocab': inv_target_vocab,
            'max_len': max_len
        }

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return create_test_dataset(max_len=max_len)

def create_test_dataset(max_len=50):
    """Create a small test dataset for debugging purposes"""
    print("Creating a minimal test dataset for debugging...")
    
    # Example data with attestation counts
    source_texts = ["hello", "world", "test", "longerword", "sampledata", "another"]
    target_texts = ["ஹலோ", "உலகம்", "சோதனை", "நீண்டசொல்", "மாதிரிதரவு", "மற்றொன்று"]
    attestation_counts = [2, 1, 3, 1, 2, 1]  # Example attestation counts

    # Ensure max_len is at least long enough for the longest example + SOS/EOS
    min_required_len = max(max(len(s) for s in source_texts), max(len(t) for t in target_texts)) + 2
    current_max_len = max(max_len, min_required_len)
    if current_max_len > max_len:
        print(f"Warning: Adjusted max_len from {max_len} to {current_max_len} for test data.")
        max_len = current_max_len

    source_chars = set("".join(source_texts))
    target_chars = set("".join(target_texts))

    # Create vocab with consistent special tokens
    source_vocab = {}
    target_vocab = {}
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    for i, token in enumerate(special_tokens):
        source_vocab[token] = i
        target_vocab[token] = i

    for char in sorted(list(source_chars)):
        if char not in source_vocab:
            source_vocab[char] = len(source_vocab)

    for char in sorted(list(target_chars)):
        if char not in target_vocab:
            target_vocab[char] = len(target_vocab)

    inv_source_vocab = {v: k for k, v in source_vocab.items()}
    inv_target_vocab = {v: k for k, v in target_vocab.items()}

    # Split into train, dev, test (using fixed small splits)
    train_src = source_texts[:4]
    train_tgt = target_texts[:4]
    train_att = attestation_counts[:4]
    
    dev_src = source_texts[4:5]
    dev_tgt = target_texts[4:5]
    dev_att = attestation_counts[4:5]
    
    test_src = source_texts[5:]
    test_tgt = target_texts[5:]
    test_att = attestation_counts[5:]

    train_dataset = TransliterationDataset(train_src, train_tgt, train_att, source_vocab, target_vocab, max_len=max_len)
    dev_dataset = TransliterationDataset(dev_src, dev_tgt, dev_att, source_vocab, target_vocab, max_len=max_len)
    test_dataset = TransliterationDataset(test_src, test_tgt, test_att, source_vocab, target_vocab, max_len=max_len)

    print("Created minimal test dataset with:")
    print(f"Train set: {len(train_dataset)} examples")
    print(f"Dev set: {len(dev_dataset)} examples")
    print(f"Test set: {len(test_dataset)} examples")
    print(f"Source vocabulary size: {len(source_vocab)}")
    print(f"Target vocabulary size: {len(target_vocab)}")
    print(f"Max sequence length: {max_len}")

    return {
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
        'test_dataset': test_dataset,
        'source_vocab': source_vocab,
        'target_vocab': target_vocab,
        'inv_source_vocab': inv_source_vocab,
        'inv_target_vocab': inv_target_vocab,
        'max_len': max_len
    }

def get_dataloaders(data_dict, batch_size=32):
    """Create DataLoaders for train, dev, and test sets"""
    train_loader = DataLoader(
        data_dict['train_dataset'],
        batch_size=batch_size,
        shuffle=True,
    )

    dev_loader = DataLoader(
        data_dict['dev_dataset'],
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        data_dict['test_dataset'],
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, dev_loader, test_loader
