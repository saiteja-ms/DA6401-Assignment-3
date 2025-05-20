import json
import matplotlib.pyplot as plt
import numpy as np
import random
import os

def create_connectivity_visualization_transliterated(predictions_file, num_samples=3, output_file='connectivity_visualization.png'):
    """
    Creates a connectivity visualization using the Latin characters instead of Tamil.
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
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(samples), 1, figsize=(12, 4 * len(samples)), dpi=100)
    
    # Handle case with only one sample
    if len(samples) == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples):
        source = sample['source']  # Latin characters (source)
        prediction = sample['prediction']  # Tamil characters (prediction)
        target = sample['target']  # Tamil characters (target)
        is_correct = sample['correct']
        attention_weights = np.array(sample['attention_weights'])
        
        ax = axes[i]
        
        # Plot source characters at the bottom (these are already Latin)
        for j, char in enumerate(source):
            ax.text(j, 0, char, ha='center', va='center', fontsize=14, 
                   bbox=dict(facecolor='lightblue', alpha=0.3, boxstyle='round'))
        
        # Plot prediction characters at the top (use indices instead of Tamil characters)
        for j in range(len(prediction)):
            if j < len(attention_weights):
                # Use index number instead of Tamil character
                ax.text(j, 1, f"[{j}]", ha='center', va='center', fontsize=14,
                       bbox=dict(facecolor='lightgreen' if is_correct else 'lightcoral', 
                                alpha=0.3, boxstyle='round'))
        
        # Draw connections based on attention weights
        for j, weights in enumerate(attention_weights):
            if j >= len(prediction):
                continue
                
            # Find the source character with highest attention
            max_idx = np.argmax(weights)
            
            # Draw a strong line for the highest attention
            ax.plot([max_idx, j], [0.1, 0.9], 'r-', alpha=0.7, linewidth=2)
            
            # Draw fainter lines for other significant attentions
            for k, weight in enumerate(weights):
                if k != max_idx and weight > 0.1:  # Threshold for significant attention
                    ax.plot([k, j], [0.1, 0.9], 'b-', alpha=0.3 * weight, linewidth=1)
        
        # Set axis limits
        ax.set_xlim(-1, max(len(source), len(prediction)))
        ax.set_ylim(-0.2, 1.2)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add labels
        ax.text(-0.05, 0, 'Source', ha='right', va='center', fontsize=12, fontweight='bold')
        ax.text(-0.05, 1, 'Target', ha='right', va='center', fontsize=12, fontweight='bold')
        
        # Add title with both source and transliterated prediction
        title_color = 'green' if is_correct else 'red'
        title_prefix = "✓" if is_correct else "✗"
        ax.set_title(f"{title_prefix} {source} → [Tamil output] (Correct: {is_correct})", 
                    color=title_color, fontsize=14, fontweight='bold')
        
        # Add a legend explaining the Tamil characters
        legend_text = "Tamil output characters:\n"
        for j, char in enumerate(prediction):
            if j < len(attention_weights):
                legend_text += f"[{j}]: {char}  "
                if (j+1) % 5 == 0:  # Line break every 5 characters
                    legend_text += "\n"
        
        ax.text(0.5, -0.15, legend_text, ha='center', va='top', fontsize=10,
               transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Connectivity visualization saved to {output_file}")
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create connectivity visualization with matplotlib')
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to the JSON file containing predictions with attention weights')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to display (default: 3)')
    parser.add_argument('--output_file', type=str, default='connectivity_visualization.png',
                        help='Path to save the PNG visualization')
    
    args = parser.parse_args()
    
    create_connectivity_visualization_transliterated(
        args.predictions_file,
        args.num_samples,
        args.output_file
    )
