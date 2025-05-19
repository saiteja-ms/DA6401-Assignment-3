import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TransliterationDataset(Dataset):
    def __init__(self, source_texts, target_texts, source_vocab, target_vocab, max_len=50):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # Convert non-string values to strings
        if not isinstance(source_text, str):
            source_text = str(source_text)
        if not isinstance(target_text, str):
            target_text = str(target_text)
        
        # Convert characters to indices
        source_indices = [self.source_vocab.get(char, self.source_vocab['<UNK>']) for char in source_text]
        target_indices = [self.target_vocab.get(char, self.target_vocab['<UNK>']) for char in target_text]
        
        # Handle different possible SOS token names
        sos_token = None
        for token in ['<SOS>', '< SOS >', 'SOS']:
            if token in self.target_vocab:
                sos_token = token
                break
        
        if sos_token is None:
            # If SOS token not found, use the first special token (usually PAD)
            sos_idx = 2  # Default SOS index
        else:
            sos_idx = self.target_vocab[sos_token]
        
        # Handle different possible EOS token names
        eos_token = None
        for token in ['<EOS>', '< EOS >', 'EOS']:
            if token in self.target_vocab:
                eos_token = token
                break
        
        if eos_token is None:
            # If EOS token not found, use index 3 (typical EOS index)
            eos_idx = 3  # Default EOS index
        else:
            eos_idx = self.target_vocab[eos_token]
        
        # Add SOS and EOS tokens to target using indices directly
        target_indices = [sos_idx] + target_indices + [eos_idx]
        
        # Pad sequences
        source_indices = source_indices[:self.max_len]
        target_indices = target_indices[:self.max_len]
        
        source_indices += [self.source_vocab['<PAD>']] * (self.max_len - len(source_indices))
        target_indices += [self.target_vocab['<PAD>']] * (self.max_len - len(target_indices))
        
        return {
            'source': torch.tensor(source_indices, dtype=torch.long),
            'target': torch.tensor(target_indices, dtype=torch.long),
            'source_text': source_text,
            'target_text': target_text
        }

def load_dakshina_data(language='ta', base_dir='.'):
    """
    Load data from the Dakshina dataset for a specific language
    language: language code (e.g., 'ta' for Tamil)
    base_dir: base directory containing the dataset
    """
    # Define file paths based on Dakshina dataset structure
    train_file = os.path.join(language, "lexicons", f"{language}.translit.sampled.train.tsv")
    dev_file = os.path.join(language, "lexicons", f"{language}.translit.sampled.dev.tsv")
    test_file = os.path.join(language, "lexicons", f"{language}.translit.sampled.test.tsv")

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
        return create_test_dataset()
    
    # Load training data with error handling
    try:
        # Use on_bad_lines='skip' to skip problematic rows
        train_df = pd.read_csv(train_file, sep='\t', header=None, on_bad_lines='skip')
        dev_df = pd.read_csv(dev_file, sep='\t', header=None, on_bad_lines='skip')
        test_df = pd.read_csv(test_file, sep='\t', header=None, on_bad_lines='skip')
        
        # Ensure we only have 2 columns and name them
        if train_df.shape[1] > 2:
            train_df = train_df.iloc[:, :2]  # Take only the first two columns
        train_df.columns = ['source', 'target']
        
        if dev_df.shape[1] > 2:
            dev_df = dev_df.iloc[:, :2]
        dev_df.columns = ['source', 'target']
        
        if test_df.shape[1] > 2:
            test_df = test_df.iloc[:, :2]
        test_df.columns = ['source', 'target']
        
        # Convert any non-string values to strings
        train_df['source'] = train_df['source'].astype(str)
        train_df['target'] = train_df['target'].astype(str)
        dev_df['source'] = dev_df['source'].astype(str)
        dev_df['target'] = dev_df['target'].astype(str)
        test_df['source'] = test_df['source'].astype(str)
        test_df['target'] = test_df['target'].astype(str)
        
        # Build vocabularies
        source_chars = set()
        target_chars = set()
        
        for text in train_df['source']:
            if isinstance(text, str):
                source_chars.update(text)
        
        for text in train_df['target']:
            if isinstance(text, str):
                target_chars.update(text)
        
        # Create vocabulary dictionaries with special tokens - UPDATED to use consistent token names
        source_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        target_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        for i, char in enumerate(sorted(source_chars)):
            source_vocab[char] = i + 4
        
        for i, char in enumerate(sorted(target_chars)):
            target_vocab[char] = i + 4
        
        # Print debug information about the vocabularies
        print("Special tokens in vocabulary:")
        print(f"Source vocab keys: {list(source_vocab.keys())[:10]}")
        print(f"Target vocab keys: {list(target_vocab.keys())[:10]}")
        
        # Create inverse vocabularies for decoding
        inv_source_vocab = {v: k for k, v in source_vocab.items()}
        inv_target_vocab = {v: k for k, v in target_vocab.items()}
        
        # Create datasets
        train_dataset = TransliterationDataset(
            train_df['source'].tolist(), 
            train_df['target'].tolist(),
            source_vocab, 
            target_vocab
        )
        
        dev_dataset = TransliterationDataset(
            dev_df['source'].tolist(), 
            dev_df['target'].tolist(),
            source_vocab, 
            target_vocab
        )
        
        test_dataset = TransliterationDataset(
            test_df['source'].tolist(), 
            test_df['target'].tolist(),
            source_vocab, 
            target_vocab
        )
        
        print(f"Successfully loaded Dakshina dataset for {language}")
        print(f"Train set: {len(train_dataset)} examples")
        print(f"Dev set: {len(dev_dataset)} examples")
        print(f"Test set: {len(test_dataset)} examples")
        print(f"Source vocabulary size: {len(source_vocab)}")
        print(f"Target vocabulary size: {len(target_vocab)}")
        
        return {
            'train_dataset': train_dataset,
            'dev_dataset': dev_dataset,
            'test_dataset': test_dataset,
            'source_vocab': source_vocab,
            'target_vocab': target_vocab,
            'inv_source_vocab': inv_source_vocab,
            'inv_target_vocab': inv_target_vocab
        }
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return create_test_dataset()

def create_test_dataset():
    """Create a small test dataset for debugging purposes"""
    print("Creating a test dataset for debugging...")
    source_texts = ["hello", "world", "this", "is", "a", "test", "sample", "data", "for", "debugging"]
    target_texts = ["ஹலோ", "உலகம்", "இது", "இஸ்", "ஒரு", "சோதனை", "மாதிரி", "தரவு", "க்கு", "பிழைத்திருத்தம்"]
    
    source_chars = set("".join(source_texts))
    target_chars = set("".join(target_texts))
    
    # Use consistent token names
    source_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    target_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    
    for i, char in enumerate(sorted(source_chars)):
        source_vocab[char] = i + 4
    
    for i, char in enumerate(sorted(target_chars)):
        target_vocab[char] = i + 4
    
    inv_source_vocab = {v: k for k, v in source_vocab.items()}
    inv_target_vocab = {v: k for k, v in target_vocab.items()}
    
    # Split into train, dev, test
    train_src = source_texts[:7]
    train_tgt = target_texts[:7]
    dev_src = source_texts[7:9]
    dev_tgt = target_texts[7:9]
    test_src = source_texts[9:]
    test_tgt = target_texts[9:]
    
    train_dataset = TransliterationDataset(train_src, train_tgt, source_vocab, target_vocab)
    dev_dataset = TransliterationDataset(dev_src, dev_tgt, source_vocab, target_vocab)
    test_dataset = TransliterationDataset(test_src, test_tgt, source_vocab, target_vocab)
    
    print("Created test dataset with:")
    print(f"Train set: {len(train_dataset)} examples")
    print(f"Dev set: {len(dev_dataset)} examples")
    print(f"Test set: {len(test_dataset)} examples")
    
    return {
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
        'test_dataset': test_dataset,
        'source_vocab': source_vocab,
        'target_vocab': target_vocab,
        'inv_source_vocab': inv_source_vocab,
        'inv_target_vocab': inv_target_vocab
    }

def get_dataloaders(data_dict, batch_size=32):
    """Create DataLoaders for train, dev, and test sets"""
    train_loader = DataLoader(
        data_dict['train_dataset'], 
        batch_size=batch_size, 
        shuffle=True
    )
    
    dev_loader = DataLoader(
        data_dict['dev_dataset'], 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        data_dict['test_dataset'], 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, dev_loader, test_loader

def try_alternative_paths(language='ta', base_dir='.'):
    """Try different possible file paths for the dataset"""
    possible_paths = [
        # Standard Dakshina paths
        os.path.join(base_dir, language, "lexicons", f"{language}.translit.sampled.train.tsv"),
        # Alternative paths
        os.path.join(base_dir, language, "lexicons", f"{language}.romanized.txt"),
        os.path.join(base_dir, language, "romanized", f"{language}.romanized.rejoined.tsv"),
        os.path.join(base_dir, f"{language}.translit.sampled.train.tsv"),
        os.path.join(base_dir, "lexicons", f"{language}.translit.sampled.train.tsv")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found dataset at: {path}")
            return os.path.dirname(path)
    
    print("Could not find dataset in any of the expected locations")
    return None

if __name__ == "__main__":
    # Example usage
    data_dict = load_dakshina_data(language='ta')
    if data_dict:
        train_loader, dev_loader, test_loader = get_dataloaders(data_dict, batch_size=32)
        
        # Sample batch
        for batch in train_loader:
            print(f"Source shape: {batch['source'].shape}")
            print(f"Target shape: {batch['target'].shape}")
            print(f"Sample source text: {batch['source_text'][0]}")
            print(f"Sample target text: {batch['target_text'][0]}")
            break
