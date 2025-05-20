import os
import argparse
import json
import wandb
import torch

def main():
    """Main function to parse arguments and run modes (train, sweep, analyze)."""
    parser = argparse.ArgumentParser(description='Transliteration System')
    parser.add_argument('--mode', type=str, choices=['train', 'sweep', 'analyze'], required=False,
                    help='Mode to run: train, sweep, or analyze')

    parser.add_argument('--language', type=str, default='ta',
                        help='Language code (e.g., ta for Tamil). Default: ta.')
    parser.add_argument('--model_type', type=str, choices=['vanilla', 'attention'], default='vanilla',
                        help='Model type: vanilla or attention. Default: vanilla.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (JSON). If None, uses default config based on model_type.')
    parser.add_argument('--data_dir', type=str, default='.', 
                        help='Base directory containing the dataset (e.g., "." if "language_code/lexicons/" is in CWD). Default: "."')
    parser.add_argument('--sweep_count', type=int, default=50,
                        help='Number of runs for hyperparameter sweep. Default: 50.')
    parser.add_argument('--vanilla_run_id', type=str, default=None,
                        help='Wandb run ID for the vanilla model predictions file (for analysis mode).')
    parser.add_argument('--attention_run_id', type=str, default=None,
                        help='Wandb run ID for the attention model predictions file (for analysis mode).')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='Maximum sequence length for padding/truncation. Default: 50.')
    parser.add_argument('--visualize', action='store_true',
                    help='Generate creative visualizations of predictions')
    parser.add_argument('--predictions_file', type=str, default=None,
                        help='Path to predictions JSON file for visualization')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to include in visualization')
    parser.add_argument('--output_path', type=str, default='prediction_samples.png',
                        help='Path to save visualization output')
    parser.add_argument('--train_and_visualize', action='store_true',
                    help='Train a model with given config and generate connectivity visualizations')
    parser.add_argument('--train_compare', action='store_true',
                    help='Train both vanilla and attention models with best configs and compare them')
    parser.add_argument('--vanilla_config', type=str, default='best_vanilla_config.json',
                        help='Config file for vanilla model')
    parser.add_argument('--attention_config', type=str, default='best_attention_config.json',
                        help='Config file for attention model')

    args = parser.parse_args()

    # Set device if needed globally for analysis visualization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        # Set CUDA flags for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("No GPU available, using CPU instead")

    # if args.train_and_visualize:
    #     from src.training.train import train_model
    #     from src.visualization.analysis import visualize_connectivity
    #     from src.data.preprocessing import load_dakshina_data, get_dataloaders
        
    #     # Check if config file exists
    #     if not args.config:
    #         print("Error: --config is required for train_and_visualize mode")
    #         return
        
    #     # Check if the config file exists in the current directory
    #     if not os.path.exists(args.config):
    #         print(f"Error: Config file '{args.config}' not found in the current directory.")
    #         print(f"Current directory: {os.getcwd()}")
    #         print(f"Files in current directory: {os.listdir('.')}")
    #         return
        
    #     # Load config
    #     try:
    #         with open(args.config, 'r') as f:
    #             config = json.load(f)
    #             print(f"Successfully loaded config from {args.config}")
    #     except Exception as e:
    #         print(f"Error loading config file: {e}")
    #         return
        
    #     # Make sure attention is enabled
    #     config['use_attention'] = True
        
    #     # Add data_dir to config
    #     config['data_dir'] = args.data_dir
    #     config['language'] = args.language
    #     config['max_seq_len'] = args.max_seq_len
        
    #     # Train model
    #     print(f"Training attention model with config: {config}")
    #     model, accuracy, predictions = train_model(config)
        
    #     if model is None:
    #         print("Error: Model training failed")
    #         return
        
    #     print(f"Model trained successfully with accuracy: {accuracy:.4f}")
        
    #     # Load data for visualization
    #     data_dict = load_dakshina_data(
    #         language=config['language'],
    #         base_dir=config['data_dir'],
    #         max_len=config['max_seq_len']
    #     )
        
    #     # Create test loader with batch size 1 for clearer visualization
    #     _, _, test_loader = get_dataloaders(data_dict, batch_size=1)
        
    #     # Generate connectivity visualizations
    #     print("Generating connectivity visualizations...")
    #     visualize_connectivity(model, test_loader, data_dict, device)
        
    #     print("Training and visualization complete")
    #     return

    if args.train_compare:
        from src.training.train import train_model
        from src.visualization.analysis import analyze_errors, compare_models, visualize_attention
        
        # Check if config files exist
        if not os.path.exists(args.vanilla_config) or not os.path.exists(args.attention_config):
            print(f"Error: Config files not found. Vanilla: {args.vanilla_config}, Attention: {args.attention_config}")
            return
        
        # Load vanilla config
        with open(args.vanilla_config, 'r') as f:
            vanilla_config = json.load(f)
        
        # Load attention config
        with open(args.attention_config, 'r') as f:
            attention_config = json.load(f)
        
        # Set common parameters
        vanilla_config['language'] = args.language
        vanilla_config['data_dir'] = args.data_dir
        vanilla_config['max_seq_len'] = args.max_seq_len
        
        attention_config['language'] = args.language
        attention_config['data_dir'] = args.data_dir
        attention_config['max_seq_len'] = args.max_seq_len
        
        # Ensure attention is set correctly
        vanilla_config['use_attention'] = False
        attention_config['use_attention'] = True
        
        # Train vanilla model
        print("\n=== Training Vanilla Model ===")
        vanilla_model, vanilla_accuracy, vanilla_predictions = train_model(vanilla_config)
        
        if vanilla_model is None:
            print("Error: Vanilla model training failed")
            return
        
        vanilla_run_id = wandb.run.id
        print(f"Vanilla model trained with run ID: {vanilla_run_id}")
        print(f"Vanilla model test accuracy: {vanilla_accuracy:.4f}")
        
        # Finish the current wandb run
        wandb.run.finish()
        
        # Train attention model
        print("\n=== Training Attention Model ===")
        attention_model, attention_accuracy, attention_predictions = train_model(attention_config)
        
        if attention_model is None:
            print("Error: Attention model training failed")
            return
        
        attention_run_id = wandb.run.id
        print(f"Attention model trained with run ID: {attention_run_id}")
        print(f"Attention model test accuracy: {attention_accuracy:.4f}")
        
        # Finish the current wandb run
        wandb.run.finish()
        
        # Start a new run for analysis
        wandb.init(project="transliteration-seq2seq", name="Models_Comparison", job_type="analysis")
        
        # Compare models
        print("\n=== Comparing Models ===")
        vanilla_file = f'predictions/vanilla/predictions-{vanilla_run_id}.json'
        attention_file = f'predictions/attention/predictions-{attention_run_id}.json'
        
        if os.path.exists(vanilla_file) and os.path.exists(attention_file):
            # Analyze individual models
            vanilla_acc, _ = analyze_errors(vanilla_file)
            attention_acc, _ = analyze_errors(attention_file)
            
            # Compare models
            compare_models(vanilla_file, attention_file)
            
            # Generate attention heatmaps
            data_dict = load_dakshina_data(
                language=args.language,
                base_dir=args.data_dir,
                max_len=args.max_seq_len
            )
            
            _, _, test_loader = get_dataloaders(data_dict, batch_size=3)
            
            visualize_attention(attention_model, test_loader, data_dict, device, attention_run_id)
            
            print(f"\nComparison complete. Vanilla accuracy: {vanilla_acc:.4f}, Attention accuracy: {attention_acc:.4f}")
        else:
            print(f"Error: Prediction files not found. Vanilla: {vanilla_file}, Attention: {attention_file}")
        
        # Finish the analysis run
        wandb.run.finish()
        
        print("\nTraining and comparison complete.")
        print(f"Vanilla Run ID: {vanilla_run_id}, Accuracy: {vanilla_accuracy:.4f}")
        print(f"Attention Run ID: {attention_run_id}, Accuracy: {attention_accuracy:.4f}")

    if args.mode == 'train':
        from src.training.train import train_model

        # Load or create config dictionary
        config = {}
        if args.config and os.path.exists(args.config):
            print(f"Loading config from {args.config}")
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            # Use default config path based on model type
            default_config_file = f"configs/{'attention' if args.model_type == 'attention' else 'vanilla'}_config.json"
            if os.path.exists(default_config_file):
                 print(f"Loading default config from {default_config_file}")
                 with open(default_config_file, 'r') as f:
                     config = json.load(f)
            else:
                 print(f"Error: Default config file not found at {default_config_file}")
                 # Create a minimal default config dictionary if file is missing
                 config = {
                     "embedding_size": 128, "hidden_size": 256,
                     "encoder_layers": 3, "decoder_layers": 3,
                     "cell_type": "lstm", "dropout": 0.3,
                     "learning_rate": 0.001, "batch_size": 128, "n_epochs": 10,
                     "clip": 1.0, "teacher_forcing_ratio": 0.5
                 }
                 print("Using minimal hardcoded default config.")

        # Override config with command line arguments (explicitly setting these from args)
        # Command line arguments have the highest priority
        config['language'] = args.language
        config['use_attention'] = args.model_type == 'attention' # Ensure consistency
        config['data_dir'] = args.data_dir # This is the crucial part, takes value from args
        config['max_seq_len'] = args.max_seq_len

        print(f"Resolved data directory passed to train_model: {config.get('data_dir', 'Not Set')}")
        print(f"Running training with resolved config: {config}")

        # Run training
        model, accuracy, predictions = train_model(config)

        if model is not None:
             print(f"Training completed. Final test accuracy: {accuracy:.4f}")
        else:
            print("Training failed.")

    elif args.mode == 'sweep':
        from src.training.train import run_sweep

        # Load or create sweep config
        if args.config and os.path.exists(args.config):
            print(f"Loading sweep config from {args.config}")
            with open(args.config, 'r') as f:
                sweep_config = json.load(f)
            # Ensure the essential fixed parameters from args are in the config if loading from file
            # This prevents the agent from trying to sweep over them if they aren't defined in the file
            sweep_config['parameters']['language'] = {'value': args.language}
            sweep_config['parameters']['data_dir'] = {'value': args.data_dir}
            sweep_config['parameters']['max_seq_len'] = {'value': args.max_seq_len}
            sweep_config['parameters']['use_attention'] = {'value': args.model_type == 'attention'}

        else:
            print("No sweep config file provided or found. Using default sweep configuration.")
            # Default sweep config with optimizer options
            sweep_config = {
                'method': 'bayes',
                'metric': {'name': 'valid_accuracy', 'goal': 'maximize'},
                'parameters': {
                    # Fixed parameters for the sweep (taken from command line args)
                    'language': {'value': args.language},
                    'data_dir': {'value': args.data_dir},
                    'max_seq_len': {'value': args.max_seq_len},
                    'use_attention': {'value': args.model_type == 'attention'},
                    
                    # Model architecture parameters
                    'embedding_size': {'values': [32, 64, 128]},
                    'hidden_size': {'values': [64, 128, 256]},
                    'num_layers': {'values': [1, 2, 3]},  # Single parameter for both encoder and decoder layers
                    'cell_type': {'values': ['rnn', 'lstm', 'gru']},
                    'dropout': {'values': [0.0, 0.2, 0.3]},
                    
                    # Optimizer parameters
                    'optimizer': {'values': ['adam', 'rmsprop', 'sgd']},
                    'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 0.01},
                    'weight_decay': {'values': [0, 0.0001, 0.001]},
                    
                    # Optimizer-specific parameters
                    'rmsprop_alpha': {'value': 0.99},
                    'momentum': {'values': [0, 0.9]},  # For SGD
                    
                    # Learning rate scheduler
                    'scheduler': {'values': ['none', 'plateau', 'cosine']},
                    'scheduler_factor': {'value': 0.5},  # For plateau
                    'scheduler_patience': {'value': 5},  # For plateau
                    'scheduler_t_max': {'value': 20},    # For cosine
                    
                    # Training parameters
                    'batch_size': {'values': [32, 64, 128]},
                    'n_epochs': {'value': 10},
                    'clip': {'value': 1.0},
                    'teacher_forcing_ratio': {'values': [0.5, 0.7]}
                }
            }


        # Run the sweep
        run_sweep(sweep_config, args.sweep_count)
        
    elif args.mode == 'analyze':
        from src.visualization.analysis import analyze_errors, compare_models, visualize_attention, visualize_connectivity
        from src.data.preprocessing import load_dakshina_data, get_dataloaders
        from src.models import Seq2Seq, AttentionSeq2Seq, Encoder, Decoder, AttentionDecoder

        if not args.vanilla_run_id and not args.attention_run_id:
            print("Error: At least one of --vanilla_run_id or --attention_run_id is required for analysis mode")
            return

        # Initialize wandb run for analysis
        wandb.init(project="transliteration-seq2seq", name="Analysis_Run", job_type="analysis") # Give the analysis run a specific name
        print(f"Starting analysis run: {wandb.run.id}")

        vanilla_acc = 0.0
        attention_acc = 0.0

        # --- Analyze individual models ---
        if args.vanilla_run_id:
            print(f"\nAnalyzing Vanilla Model Run ID: {args.vanilla_run_id}")
            vanilla_file = f'predictions/vanilla/predictions-{args.vanilla_run_id}.json'
            vanilla_acc, _ = analyze_errors(vanilla_file) # analyze_errors logs results internally
            if wandb.run:
                 wandb.run.log({"vanilla_test_accuracy_from_analysis": vanilla_acc}) # Log explicitly in analysis run

        if args.attention_run_id:
            print(f"\nAnalyzing Attention Model Run ID: {args.attention_run_id}")
            attention_file = f'predictions/attention/predictions-{args.attention_run_id}.json'
            attention_acc, _ = analyze_errors(attention_file) # analyze_errors logs results internally
            if wandb.run:
                 wandb.run.log({"attention_test_accuracy_from_analysis": attention_acc}) # Log explicitly in analysis run

        # --- Compare models if both are provided ---
        if args.vanilla_run_id and args.attention_run_id:
            print(f"\nComparing Vanilla ({args.vanilla_run_id}) vs Attention ({args.attention_run_id}) models")
            vanilla_file = f'predictions/vanilla/predictions-{args.vanilla_run_id}.json'
            attention_file = f'predictions/attention/predictions-{args.attention_run_id}.json'
            compare_models(vanilla_file, attention_file)

        # --- Visualize Attention/Connectivity (requires loading a model) ---
        if args.attention_run_id:
             print(f"\nAttempting to load model {args.attention_run_id} for visualization...")
             viz_data_dir = args.data_dir
             viz_max_seq_len = args.max_seq_len
             viz_language = args.language # Assume visualization is for the same language

             data_dict = load_dakshina_data(viz_language, viz_data_dir, viz_max_seq_len)
             if data_dict and data_dict.get('test_dataset') and len(data_dict['test_dataset']) > 0:
                 # Use a smaller batch size for visualization if needed due to memory
                 viz_batch_size = min(16, len(data_dict['test_dataset'])) # Don't exceed dataset size
                 if viz_batch_size == 0:
                     print("Test dataset is empty after loading, cannot visualize.")
                 else:
                     _, _, test_loader = get_dataloaders(data_dict, batch_size=viz_batch_size)

                     # Try to find the model file based on run_id
                     model_file_path = f'best-model-{args.attention_run_id}.pt'

                     if os.path.exists(model_file_path):
                          print(f"Found model file: {model_file_path}")
                          try:
                               # Load default attention config to get sizes and types
                               default_viz_config_path = "configs/attention_config.json"
                               viz_config_defaults = {}
                               if os.path.exists(default_viz_config_path):
                                    with open(default_viz_config_path, 'r') as f:
                                         viz_config_defaults = json.load(f)
                               else:
                                    print(f"Warning: Default viz config not found at {default_viz_config_path}. Using minimal hardcoded values.")
                                    # Fallback minimal values
                                    viz_config_defaults = {"embedding_size": 64, "hidden_size": 128, "encoder_layers": 1, "decoder_layers": 1, "cell_type": "lstm", "dropout": 0.2}

                               temp_config = {
                                   'embedding_size': viz_config_defaults.get('embedding_size', 64),
                                   'hidden_size': viz_config_defaults.get('hidden_size', 128),
                                   'num_layers': viz_config_defaults.get('encoder_layers', viz_config_defaults.get('decoder_layers', 1)),
                                   'cell_type': viz_config_defaults.get('cell_type', 'lstm'),
                                   'dropout': viz_config_defaults.get('dropout', 0.2),
                                   'output_size': len(data_dict['target_vocab']),
                                   'input_size': len(data_dict['source_vocab'])
                               }

                               encoder = Encoder(
                                   input_size=temp_config['input_size'],
                                   embedding_size=temp_config['embedding_size'],
                                   hidden_size=temp_config['hidden_size'],
                                   num_layers=temp_config['num_layers'],
                                   cell_type=temp_config['cell_type'],
                                   dropout=temp_config['dropout']
                               ).to(device)

                               decoder = AttentionDecoder(
                                   output_size=temp_config['output_size'],
                                   embedding_size=temp_config['embedding_size'],
                                   encoder_hidden_size=temp_config['hidden_size'],
                                   decoder_hidden_size=temp_config['hidden_size'],
                                   num_layers=temp_config['num_layers'],
                                   cell_type=temp_config['cell_type'],
                                   dropout=temp_config['dropout']
                               ).to(device)

                               model = AttentionSeq2Seq(encoder, decoder, device).to(device)

                               # Load the state dict
                               model.load_state_dict(torch.load(model_file_path, map_location=device))
                               model.eval() # Set to eval mode

                               # Generate visualizations
                               # Ensure test_loader has data before passing
                               if len(test_loader.dataset) > 0:
                                    visualize_attention(model, test_loader, data_dict, device, args.attention_run_id)
                                    visualize_connectivity(model, test_loader, data_dict, device, args.attention_run_id)
                               else:
                                    print("Test loader is empty, skipping visualization.")

                          except Exception as e:
                               print(f"Error loading model {model_file_path} or generating visualizations: {e}")
                               import traceback
                               traceback.print_exc()

                     else:
                          print(f"Model file not found at {model_file_path}. Cannot generate visualizations.")
             else:
                  print("Could not load data for visualization or test dataset is empty.")

        # Finish the wandb analysis run
        if wandb.run:
             wandb.run.finish()
        print("Analysis completed.")

if __name__ == "__main__":
    main()
