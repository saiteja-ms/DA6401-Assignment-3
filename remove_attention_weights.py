import json
import os

def remove_attention_weights(input_file, output_file=None):
    """
    Removes the 'attention_weights' key from each prediction in a JSON file.
    
    Args:
        input_file: Path to the input JSON file with predictions
        output_file: Path to save the cleaned predictions. If None, will use input_file with '_no_weights' suffix
    """
    if output_file is None:
        # Create output filename by adding '_no_weights' before the extension
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_no_weights{ext}"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    try:
        # Load the predictions
        with open(input_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        print(f"Loaded {len(predictions)} predictions from {input_file}")
        
        # Remove attention_weights from each prediction
        cleaned_count = 0
        for pred in predictions:
            if 'attention_weights' in pred:
                del pred['attention_weights']
                cleaned_count += 1
        
        # Save the cleaned predictions
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        print(f"Removed attention weights from {cleaned_count} predictions")
        print(f"Saved cleaned predictions to '{output_file}'")
        return True
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # You can run this script directly with your file path
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        remove_attention_weights(input_file, output_file)
    else:
        print("Usage: python remove_attention_weights.py input_file.json [output_file.json]")
