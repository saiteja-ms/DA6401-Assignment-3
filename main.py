import os
import argparse
import json
import wandb

def main():
    parser = argparse.ArgumentParser(description='Transliteration System')
    parser.add_argument('--mode', type=str, choices=['train', 'sweep', 'analyze'], required=True,
                        help='Mode to run: train, sweep, or analyze')
    parser.add_argument('--language', type=str, default='ta',
                        help='Language code (e.g., ta for Tamil)')
    parser.add_argument('--model_type', type=str, choices=['vanilla', 'attention'], default='vanilla',
                        help='Model type: vanilla or attention')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Base directory containing the dataset')
    parser.add_argument('--sweep_count', type=int, default=50,
                        help='Number of runs for hyperparameter sweep')
    parser.add_argument('--vanilla_run_id', type=str, default=None,
                        help='Run ID for vanilla model (for analysis)')
    parser.add_argument('--attention_run_id', type=str, default=None,
                        help='Run ID for attention model (for analysis)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        from src.training.train import train_model
        
        # Load config
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            # Default config
            config_file = f"configs/{'attention' if args.model_type == 'attention' else 'vanilla'}_config.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Update config with command line args
        config['language'] = args.language
        config['use_attention'] = args.model_type == 'attention'
        config['data_dir'] = args.data_dir
        
        model, accuracy, predictions = train_model(config)
        print(f"Model trained with accuracy: {accuracy:.4f}")
    
    elif args.mode == 'sweep':
        from src.training.train import run_sweep
        
        # Set up sweep config
        if args.config:
            with open(args.config, 'r') as f:
                sweep_config = json.load(f)
        else:
            # Default sweep config
            sweep_config = {
                'method': 'bayes',
                'metric': {'name': 'valid_accuracy', 'goal': 'maximize'},
                'parameters': {
                    'language': {'value': args.language},
                    'data_dir': {'value': args.data_dir},
                    'embedding_size': {'values': [16, 32, 64, 256]},
                    'hidden_size': {'values': [16, 32, 64, 256]},
                    'encoder_layers': {'values': [1, 2, 3]},
                    'decoder_layers': {'values': [1, 2, 3]},
                    'cell_type': {'values': ['rnn', 'lstm', 'gru']},
                    'dropout': {'values': [0.2, 0.3]},
                    'learning_rate': {'values': [0.001, 0.01, 0.1]},
                    'batch_size': {'values': [32, 64, 128]},
                    'n_epochs': {'value': 10},
                    'clip': {'value': 1.0},
                    'teacher_forcing_ratio': {'value': 0.5},
                    'use_attention': {'value': args.model_type == 'attention'}
                }
            }
        
        # Run sweep
        run_sweep(sweep_config, args.sweep_count)
    
    elif args.mode == 'analyze':
        if not args.vanilla_run_id or not args.attention_run_id:
            print("Error: Both vanilla_run_id and attention_run_id are required for analysis mode")
            return
        
        from src.visualization.analysis import analyze_errors, compare_models
        
        # Initialize wandb
        wandb.init(project="transliteration-seq2seq")
        
        # Analyze vanilla model
        vanilla_file = f'predictions/vanilla/predictions-{args.vanilla_run_id}.json'
        vanilla_accuracy, vanilla_errors = analyze_errors(vanilla_file)
        
        # Analyze attention model
        attention_file = f'predictions/attention/predictions-{args.attention_run_id}.json'
        attention_accuracy, attention_errors = analyze_errors(attention_file)
        
        # Compare models
        compare_models(vanilla_file, attention_file)
        
        # Finish wandb run
        wandb.finish()

if __name__ == "__main__":
    main()
