import json
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.font_manager as fm
import os
from matplotlib import rcParams
import seaborn as sns

def setup_tamil_font():
    """
    Configure matplotlib to use a font that supports Tamil characters.
    Returns True if successful, False otherwise.
    """
    # First, try to find Tamil fonts on the system
    tamil_fonts = [f.name for f in fm.fontManager.ttflist 
                  if any(tamil_name in f.name.lower() 
                        for tamil_name in ['tamil', 'latha', 'nirmala', 'noto'])]
    
    print("Potential Tamil-compatible fonts found:", tamil_fonts)
    
    # Common fonts that support Tamil
    potential_tamil_fonts = ['Latha', 'Nirmala UI', 'Noto Sans Tamil', 'Tamil MN', 
                            'InaiMathi', 'Vijaya', 'Akshar Unicode', 'Arial Unicode MS']
    
    # Add any found Tamil fonts to our list
    potential_tamil_fonts.extend(tamil_fonts)
    
    # Try to set a Tamil font
    for font in potential_tamil_fonts:
        try:
            plt.rcParams['font.family'] = font
            print(f"Using font: {font}")
            return True
        except:
            continue
    
    # If no Tamil font works, try setting a generic Unicode font
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("Using DejaVu Sans as fallback font")
        return True
    except:
        print("Warning: Could not set a suitable font for Tamil text")
        return False

def create_creative_prediction_grid(predictions_file, num_samples=9, output_file='prediction_grid.png'):
    """
    Creates a visually appealing grid visualization of sample predictions.
    
    Args:
        predictions_file: Path to the JSON file containing predictions
        num_samples: Number of samples to display (preferably a square number)
        output_file: Path to save the output image
    """
    # Setup font for Tamil text
    font_success = setup_tamil_font()
    
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Get some correct and incorrect examples
    correct_preds = [p for p in predictions if p['correct']]
    incorrect_preds = [p for p in predictions if not p['correct']]
    
    # Sample predictions
    num_correct = min(num_samples // 2, len(correct_preds))
    num_incorrect = min(num_samples - num_correct, len(incorrect_preds))
    
    sampled_correct = random.sample(correct_preds, num_correct) if correct_preds else []
    sampled_incorrect = random.sample(incorrect_preds, num_incorrect) if incorrect_preds else []
    
    samples = sampled_correct + sampled_incorrect
    random.shuffle(samples)
    
    # Calculate grid dimensions (try to make it square)
    grid_size = int(np.ceil(np.sqrt(len(samples))))
    
    # Create figure and axes
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15), dpi=100)
    
    # Flatten axes for easier indexing
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Custom color palette
    correct_colors = ['#e6ffe6', '#d9f2d9', '#ccffcc']  # Shades of green
    incorrect_colors = ['#ffe6e6', '#ffcccc', '#ffb3b3']  # Shades of red
    
    for i, sample in enumerate(samples):
        if i >= len(axes):
            break
            
        source = sample['source']
        target = sample['target']
        prediction = sample['prediction']
        is_correct = sample['correct']
        attestation = sample.get('attestation', 1)
        
        # Create text display
        if font_success:
            # If we have a Tamil font, show the actual text
            text = f"Source: {source}\nTarget: {target}\nPrediction: {prediction}"
        else:
            # If no Tamil font, show source and use placeholders for Tamil
            text = f"Source: {source}\nTarget: [Tamil Text]\nPrediction: [Tamil Text]"
            # Add a note about the actual values
            axes[i].set_xlabel(f"Target: {len(target)} chars, Prediction: {len(prediction)} chars", 
                              fontsize=8)
        
        # Set background color based on correctness with some variation
        if is_correct:
            bg_color = correct_colors[i % len(correct_colors)]
            border_color = 'green'
            title_prefix = "✓"
        else:
            bg_color = incorrect_colors[i % len(incorrect_colors)]
            border_color = 'red'
            title_prefix = "✗"
        
        # Create a text box with the prediction info
        props = dict(boxstyle='round,pad=1', facecolor=bg_color, alpha=0.7)
        
        # Add a title with attestation count
        axes[i].set_title(f"{title_prefix} Attestation: {attestation}", fontsize=12, 
                         color='darkgreen' if is_correct else 'darkred',
                         fontweight='bold')
        
        # Remove axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Add the text box
        axes[i].text(0.5, 0.5, text, transform=axes[i].transAxes, fontsize=12,
                    verticalalignment='center', horizontalalignment='center', 
                    bbox=props, fontweight='bold')
        
        # Add a border
        for spine in axes[i].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
    
    # Hide any unused subplots
    for j in range(len(samples), len(axes)):
        axes[j].axis('off')
    
    # Add a title to the entire figure
    plt.suptitle("Sample Predictions from Test Data", fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the title
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Visualization saved to {output_file}")
    return fig

def create_html_prediction_grid(predictions_file, num_samples=9, output_file='prediction_grid.html'):
    """
    Creates an HTML file with a grid of predictions that properly displays Tamil text.
    This is a fallback method if matplotlib cannot render Tamil characters.
    """
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Get some correct and incorrect examples
    correct_preds = [p for p in predictions if p['correct']]
    incorrect_preds = [p for p in predictions if not p['correct']]
    
    # Sample predictions
    num_correct = min(num_samples // 2, len(correct_preds))
    num_incorrect = min(num_samples - num_correct, len(incorrect_preds))
    
    sampled_correct = random.sample(correct_preds, num_correct) if correct_preds else []
    sampled_incorrect = random.sample(incorrect_preds, num_incorrect) if incorrect_preds else []
    
    samples = sampled_correct + sampled_incorrect
    random.shuffle(samples)
    
    # Create HTML content
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { text-align: center; margin-bottom: 20px; }
            .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
            .cell { border: 3px solid; padding: 15px; border-radius: 10px; }
            .correct { border-color: green; background-color: #e6ffe6; }
            .incorrect { border-color: red; background-color: #ffe6e6; }
            .title { font-weight: bold; margin-bottom: 10px; }
            .field { margin: 5px 0; }
            .label { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Sample Predictions from Test Data</h1>
        <div class="grid">
    """
    
    for sample in samples:
        source = sample['source']
        target = sample['target']
        prediction = sample['prediction']
        is_correct = sample['correct']
        attestation = sample.get('attestation', 1)
        
        cell_class = "correct" if is_correct else "incorrect"
        status_mark = "✓" if is_correct else "✗"
        
        html += f"""
        <div class="cell {cell_class}">
            <div class="title">{status_mark} Attestation: {attestation}</div>
            <div class="field"><span class="label">Source:</span> {source}</div>
            <div class="field"><span class="label">Target:</span> {target}</div>
            <div class="field"><span class="label">Prediction:</span> {prediction}</div>
        </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML visualization saved to {output_file}")
    return output_file

def visualize_attention_heatmaps(predictions_file, num_samples=9, output_file='attention_heatmaps.png'):
    """
    Creates a grid of attention heatmaps from predictions that include attention weights.
    
    Args:
        predictions_file: Path to the JSON file containing predictions with attention weights
        num_samples: Number of samples to display (preferably a square number)
        output_file: Path to save the output image
    """
    # Setup font for Tamil text
    font_success = setup_tamil_font()
    
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
    
    # Calculate grid dimensions (try to make it square)
    grid_size = int(np.ceil(np.sqrt(len(samples))))
    
    # Create figure and axes
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15), dpi=100)
    
    # Flatten axes for easier indexing
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, sample in enumerate(samples):
        if i >= len(axes):
            break
            
        source = sample['source']
        target = sample['target']
        prediction = sample['prediction']
        is_correct = sample['correct']
        attention_weights = sample['attention_weights']
        
        # Convert attention weights to numpy array
        attn_matrix = np.array(attention_weights)
        
        # Create heatmap
        sns.heatmap(
            attn_matrix, 
            ax=axes[i],
            cmap='viridis',
            xticklabels=list(source),
            yticklabels=list(prediction),
            cbar=False
        )
        
        # Set title
        title_color = 'green' if is_correct else 'red'
        title_prefix = "✓" if is_correct else "✗"
        axes[i].set_title(f"{title_prefix} {source} → {prediction}", 
                         color=title_color, fontsize=10)
        
        # Set labels
        axes[i].set_xlabel('Source Characters', fontsize=8)
        axes[i].set_ylabel('Predicted Characters', fontsize=8)
        
        # Rotate x-axis labels for better readability
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Hide any unused subplots
    for j in range(len(samples), len(axes)):
        axes[j].axis('off')
    
    # Add a title to the entire figure
    plt.suptitle("Attention Heatmaps (3×3 Grid)", fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the title
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Attention heatmaps saved to {output_file}")
    return fig

def create_connectivity_visualization(predictions_file, num_samples=3, output_file='connectivity_visualization.png'):
    """
    Creates a connectivity visualization similar to the one in the Distill article.
    Shows which source characters the model attends to when generating each target character.
    
    Args:
        predictions_file: Path to the JSON file containing predictions with attention weights
        num_samples: Number of samples to display
        output_file: Path to save the output image
    """
    # Setup font for Tamil text
    font_success = setup_tamil_font()
    
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
        source = sample['source']
        prediction = sample['prediction']
        is_correct = sample['correct']
        attention_weights = np.array(sample['attention_weights'])
        
        ax = axes[i]
        
        # Plot source characters at the bottom
        for j, char in enumerate(source):
            ax.text(j, 0, char, ha='center', va='center', fontsize=14, 
                   bbox=dict(facecolor='lightblue', alpha=0.3, boxstyle='round'))
        
        # Plot target characters at the top
        for j, char in enumerate(prediction):
            if j < len(attention_weights):  # Ensure we have attention weights for this character
                ax.text(j, 1, char, ha='center', va='center', fontsize=14,
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
        
        # Add title
        title_color = 'green' if is_correct else 'red'
        title_prefix = "✓" if is_correct else "✗"
        ax.set_title(f"{title_prefix} {source} → {prediction}", 
                    color=title_color, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Connectivity visualization saved to {output_file}")
    return fig

def compare_models_visualization(vanilla_file, attention_file, num_samples=5, output_file='model_comparison.png'):
    """
    Creates a visualization comparing predictions from vanilla and attention models.
    Focuses on examples where attention model corrected vanilla model's errors.
    
    Args:
        vanilla_file: Path to the JSON file with vanilla model predictions
        attention_file: Path to the JSON file with attention model predictions
        num_samples: Number of examples to display
        output_file: Path to save the output image
    """
    # Setup font for Tamil text
    font_success = setup_tamil_font()
    
    # Load predictions
    with open(vanilla_file, 'r', encoding='utf-8') as f:
        vanilla_preds = json.load(f)
    
    with open(attention_file, 'r', encoding='utf-8') as f:
        attention_preds = json.load(f)
    
    # Convert to dictionaries for easier comparison
    vanilla_dict = {p['source']: p for p in vanilla_preds}
    attention_dict = {p['source']: p for p in attention_preds}
    
    # Find examples where attention corrected vanilla
    corrected_examples = []
    for source, attn_pred in attention_dict.items():
        if source in vanilla_dict:
            vanilla_pred = vanilla_dict[source]
            if not vanilla_pred['correct'] and attn_pred['correct']:
                corrected_examples.append({
                    'source': source,
                    'target': attn_pred['target'],
                    'vanilla_pred': vanilla_pred['prediction'],
                    'attention_pred': attn_pred['prediction'],
                    'attestation': attn_pred.get('attestation', 1)
                })
    
    if not corrected_examples:
        print("No examples found where attention model corrected vanilla model's errors.")
        return None
    
    # Sample examples
    samples = random.sample(corrected_examples, min(num_samples, len(corrected_examples)))
    
    # Create figure
    fig, axes = plt.subplots(len(samples), 1, figsize=(12, 4 * len(samples)), dpi=100)
    
    # Handle case with only one sample
    if len(samples) == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples):
        source = sample['source']
        target = sample['target']
        vanilla_pred = sample['vanilla_pred']
        attention_pred = sample['attention_pred']
        attestation = sample['attestation']
        
        ax = axes[i]
        
        # Create a table-like display
        table_data = [
            ['Source', source],
            ['Target', target],
            ['Vanilla Prediction', vanilla_pred + ' ✗'],
            ['Attention Prediction', attention_pred + ' ✓']
        ]
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create a table
        table = ax.table(
            cellText=table_data,
            cellLoc='left',
            loc='center',
            cellColours=[
                ['#f2f2f2', 'white'],
                ['#f2f2f2', 'white'],
                ['#f2f2f2', '#ffe6e6'],  # Light red for incorrect vanilla prediction
                ['#f2f2f2', '#e6ffe6']   # Light green for correct attention prediction
            ]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Add a title
        ax.set_title(f"Example {i+1}: Attention Model Correction (Attestation: {attestation})", 
                    fontsize=14, fontweight='bold')
        
        # Add a border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('blue')
    
    plt.suptitle("Examples Where Attention Model Corrected Vanilla Model", 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the title
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Model comparison visualization saved to {output_file}")
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create visualizations of model predictions')
    parser.add_argument('--predictions_file', type=str, default=None,
                        help='Path to the JSON file containing predictions')
    parser.add_argument('--vanilla_file', type=str, default=None,
                        help='Path to the JSON file with vanilla model predictions')
    parser.add_argument('--attention_file', type=str, default=None,
                        help='Path to the JSON file with attention model predictions')
    parser.add_argument('--num_samples', type=int, default=9,
                        help='Number of samples to display (default: 9)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--visualization_type', type=str, 
                        choices=['grid', 'html', 'heatmap', 'connectivity', 'comparison', 'all'],
                        default='all',
                        help='Type of visualization to create')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which visualizations to create
    create_grid = args.visualization_type in ['grid', 'all']
    create_html = args.visualization_type in ['html', 'all']
    create_heatmap = args.visualization_type in ['heatmap', 'all']
    create_connectivity = args.visualization_type in ['connectivity', 'all']
    create_comparison = args.visualization_type in ['comparison', 'all']
    
    # Create visualizations
    if (create_grid or create_html) and args.predictions_file:
        if create_grid:
            try:
                create_creative_prediction_grid(
                    args.predictions_file,
                    args.num_samples,
                    os.path.join(args.output_dir, 'prediction_grid.png')
                )
            except Exception as e:
                print(f"Error creating PNG grid visualization: {e}")
        
        if create_html:
            create_html_prediction_grid(
                args.predictions_file,
                args.num_samples,
                os.path.join(args.output_dir, 'prediction_grid.html')
            )
    
    if create_heatmap and args.predictions_file:
        try:
            visualize_attention_heatmaps(
                args.predictions_file,
                args.num_samples,
                os.path.join(args.output_dir, 'attention_heatmaps.png')
            )
        except Exception as e:
            print(f"Error creating attention heatmaps: {e}")
    
    if create_connectivity and args.predictions_file:
        try:
            create_connectivity_visualization(
                args.predictions_file,
                min(3, args.num_samples),
                os.path.join(args.output_dir, 'connectivity_visualization.png')
            )
        except Exception as e:
            print(f"Error creating connectivity visualization: {e}")
    
    if create_comparison and args.vanilla_file and args.attention_file:
        try:
            compare_models_visualization(
                args.vanilla_file,
                args.attention_file,
                min(5, args.num_samples),
                os.path.join(args.output_dir, 'model_comparison.png')
            )
        except Exception as e:
            print(f"Error creating model comparison visualization: {e}")
    
    print("Visualization generation complete.")
