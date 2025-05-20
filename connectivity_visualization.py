import json
import numpy as np
import os
import random

def create_html_connectivity_visualization(predictions_file, num_samples=3, output_file='connectivity_visualization.html'):
    """
    Creates an HTML-based connectivity visualization that properly displays Tamil characters.
    """
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Filter predictions that have attention weights
    predictions_with_attn = [p for p in predictions if 'attention_weights' in p]
    
    if not predictions_with_attn:
        print("No predictions with attention weights found in the file.")
        return None
    
    # Sample predictions
    samples = random.sample(predictions_with_attn, min(num_samples, len(predictions_with_attn)))
    
    # Create HTML content
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Connectivity Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { text-align: center; }
            .example { margin-bottom: 40px; border: 1px solid #ccc; padding: 20px; border-radius: 10px; }
            .title { font-weight: bold; margin-bottom: 10px; font-size: 18px; }
            .correct { color: green; }
            .incorrect { color: red; }
            .container { position: relative; height: 200px; margin: 20px 0; }
            .source-chars { position: absolute; bottom: 0; width: 100%; display: flex; justify-content: space-around; }
            .target-chars { position: absolute; top: 0; width: 100%; display: flex; justify-content: space-around; }
            .char-box { padding: 10px; border-radius: 5px; }
            .source-char { background-color: lightblue; }
            .target-char { background-color: lightgreen; }
            svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; }
            .strong-line { stroke: red; stroke-width: 2; }
            .weak-line { stroke: blue; stroke-opacity: 0.3; stroke-width: 1; }
        </style>
    </head>
    <body>
        <h1>Connectivity Visualization</h1>
    """
    
    for i, sample in enumerate(samples):
        source = sample['source']
        prediction = sample['prediction']
        target = sample['target']
        is_correct = sample['correct']
        attention_weights = np.array(sample['attention_weights'])
        
        # Create an example container
        html += f"""
        <div class="example">
            <div class="title {'correct' if is_correct else 'incorrect'}">
                {'✓' if is_correct else '✗'} Example {i+1}: {source} → {prediction} (Target: {target})
            </div>
            <div class="container" id="container-{i}">
                <div class="source-chars">
        """
        
        # Add source characters
        for j, char in enumerate(source):
            html += f'<div class="char-box source-char" id="source-{i}-{j}">{char}</div>'
        
        html += """
                </div>
                <div class="target-chars">
        """
        
        # Add target characters
        for j, char in enumerate(prediction):
            if j < len(attention_weights):
                html += f'<div class="char-box target-char" id="target-{i}-{j}">{char}</div>'
        
        html += """
                </div>
                <svg>
        """
        
        # Add connections based on attention weights
        for j, weights in enumerate(attention_weights):
            if j >= len(prediction):
                continue
                
            # Find the source character with highest attention
            max_idx = np.argmax(weights)
            
            # Calculate positions (approximate)
            source_percent = (max_idx + 0.5) / len(source) * 100
            target_percent = (j + 0.5) / len(prediction) * 100
            
            # Add a strong line for the highest attention
            html += f'<line x1="{source_percent}%" y1="80%" x2="{target_percent}%" y2="20%" class="strong-line" />'
            
            # Add fainter lines for other significant attentions
            for k, weight in enumerate(weights):
                if k != max_idx and weight > 0.1:  # Threshold for significant attention
                    source_percent_k = (k + 0.5) / len(source) * 100
                    opacity = min(weight * 0.8, 0.8)  # Scale opacity by weight
                    html += f'<line x1="{source_percent_k}%" y1="80%" x2="{target_percent}%" y2="20%" class="weak-line" style="stroke-opacity: {opacity}" />'
        
        html += """
                </svg>
            </div>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML connectivity visualization saved to {output_file}")
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create connectivity visualization')
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to the JSON file containing predictions with attention weights')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to display (default: 3)')
    parser.add_argument('--output_file', type=str, default='connectivity_visualization.html',
                        help='Path to save the HTML visualization')
    
    args = parser.parse_args()
    
    create_html_connectivity_visualization(
        args.predictions_file,
        args.num_samples,
        args.output_file
    )
