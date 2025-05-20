import json
import numpy as np
import random
import os

def generate_compact_heatmaps_html(predictions_file, output_file='compact_heatmaps.html', num_samples=10):
    """
    Generates an HTML file with a compact 3×3 grid of attention heatmaps.
    Each heatmap has a fixed size regardless of the input length.
    
    Args:
        predictions_file: Path to the JSON file containing predictions with attention weights
        output_file: Path to save the HTML file
        num_samples: Number of samples to display (should be 9 for a 3×3 grid)
    """
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Filter predictions that have attention weights
    predictions_with_attn = [p for p in predictions if 'attention_weights' in p]
    
    if not predictions_with_attn:
        print("No predictions with attention weights found in the file.")
        return None
    
    # Filter for shorter examples (to fit better in the grid)
    shorter_examples = [p for p in predictions_with_attn 
                        if len(p['source']) <= 10 and len(p['prediction']) <= 10]
    
    # If we don't have enough shorter examples, use any examples
    if len(shorter_examples) < num_samples:
        shorter_examples = predictions_with_attn
    
    # Separate correct and incorrect predictions
    correct_preds = [p for p in shorter_examples if p['correct']]
    incorrect_preds = [p for p in shorter_examples if not p['correct']]
    
    # Sample predictions for the 3x3 grid
    num_correct = min(7, len(correct_preds))  # Aim for 7 correct, 2 incorrect
    num_incorrect = min(num_samples - num_correct, len(incorrect_preds))
    
    sampled_correct = random.sample(correct_preds, num_correct) if correct_preds else []
    sampled_incorrect = random.sample(incorrect_preds, num_incorrect) if incorrect_preds else []
    
    samples = sampled_correct + sampled_incorrect
    random.shuffle(samples)
    
    # Ensure we have exactly num_samples
    samples = samples[:num_samples]
    
    # Start building the HTML content
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Attention Heatmaps (3×3 Grid)</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-gap: 15px;
            max-width: 900px;
            margin: 0 auto;
        }
        .heatmap-container {
            background-color: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .heatmap-title {
            text-align: center;
            margin-bottom: 5px;
            font-weight: bold;
            font-size: 12px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .correct {
            color: green;
        }
        .incorrect {
            color: red;
        }
        .heatmap-canvas {
            width: 250px;
            height: 250px;
            margin: 0 auto;
            position: relative;
        }
        .legend {
            margin-top: 30px;
            text-align: center;
        }
        .legend-gradient {
            width: 300px;
            height: 20px;
            margin: 10px auto;
            background: linear-gradient(to right, #440154, #414487, #2a788e, #22a884, #7ad151, #fde725);
        }
        .legend-labels {
            display: flex;
            justify-content: space-between;
            width: 300px;
            margin: 0 auto;
        }
    </style>
    <script>
        // Function to draw a heatmap on a canvas
        function drawHeatmap(canvasId, source, target, weights, isCorrect) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Increase padding to make room for labels
            const paddingLeft = 40;   // More space for y-axis labels
            const paddingTop = 30;    // Space for title
            const paddingRight = 10;  // Right margin
            const paddingBottom = 40; // More space for x-axis labels
            
            // Calculate cell size
            const availWidth = canvas.width - (paddingLeft + paddingRight);
            const availHeight = canvas.height - (paddingTop + paddingBottom);
            const cellWidth = availWidth / source.length;
            const cellHeight = availHeight / target.length;
            
            // Draw heatmap cells first (so labels appear on top)
            for (let i = 0; i < target.length; i++) {
                for (let j = 0; j < source.length; j++) {
                    const weight = weights[i][j];
                    ctx.fillStyle = viridisColor(weight);
                    ctx.fillRect(
                        paddingLeft + j * cellWidth, 
                        paddingTop + i * cellHeight, 
                        cellWidth, 
                        cellHeight
                    );
                    
                    // Add cell border
                    ctx.strokeStyle = '#cccccc';
                    ctx.lineWidth = 0.5;
                    ctx.strokeRect(
                        paddingLeft + j * cellWidth, 
                        paddingTop + i * cellHeight, 
                        cellWidth, 
                        cellHeight
                    );
                }
            }
            
            // Draw source labels (x-axis) - BELOW the heatmap
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            for (let j = 0; j < source.length; j++) {
                // Position for bottom labels (source) - moved down further
                const labelY = paddingTop + availHeight + 20; // Positioned well below the heatmap
                
                // Add a white background behind the label
                const textWidth = ctx.measureText(source[j]).width;
                ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.fillRect(
                    paddingLeft + j * cellWidth + cellWidth/2 - textWidth/2 - 2,
                    labelY - 6, 
                    textWidth + 4, 
                    12
                );
                
                // Draw the text
                ctx.fillStyle = '#000000';
                ctx.fillText(source[j], paddingLeft + j * cellWidth + cellWidth/2, labelY);
            }
            
            // Draw target labels (y-axis)
            ctx.textAlign = 'right';
            for (let i = 0; i < target.length; i++) {
                // Add a white background behind the label
                const textWidth = ctx.measureText(target[i]).width;
                ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.fillRect(
                    paddingLeft - textWidth - 6,
                    paddingTop + i * cellHeight + cellHeight/2 - 6, 
                    textWidth + 4, 
                    12
                );
                
                // Draw the text
                ctx.fillStyle = '#000000';
                ctx.fillText(target[i], paddingLeft - 5, paddingTop + i * cellHeight + cellHeight/2);
            }
            
            // Add axis labels - positioned to avoid overlap
            ctx.textAlign = 'center';
            ctx.fillStyle = '#000000';
            ctx.font = '10px Arial';
            
            // X-axis label (at the very bottom)
            ctx.fillText("Source (English)", paddingLeft + availWidth/2, canvas.height - 5);
            
            // Y-axis label (rotated and positioned at the far left)
            ctx.save();
            ctx.translate(10, paddingTop + availHeight/2);
            ctx.rotate(-Math.PI/2);
            ctx.fillText("Target (Tamil)", 0, 0);
            ctx.restore();
        }
        
        // Function to generate a color from the viridis colormap
        function viridisColor(value) {
            // Simplified viridis colormap
            const colors = [
                [68, 1, 84],    // Dark purple (0.0)
                [65, 68, 135],   // Purple (0.2)
                [42, 120, 142],  // Blue (0.4)
                [34, 168, 132],  // Teal (0.6)
                [122, 209, 81],  // Green (0.8)
                [253, 231, 37]   // Yellow (1.0)
            ];
            
            // Find the two colors to interpolate between
            const idx = Math.min(Math.floor(value * 5), 4);
            const t = (value * 5) - idx;
            
            // Linear interpolation between the two colors
            const r = Math.round(colors[idx][0] * (1 - t) + colors[idx + 1][0] * t);
            const g = Math.round(colors[idx][1] * (1 - t) + colors[idx + 1][1] * t);
            const b = Math.round(colors[idx][2] * (1 - t) + colors[idx + 1][2] * t);
            
            return `rgb(${r}, ${g}, ${b})`;
        }
        
        // Function to initialize all heatmaps when the page loads
        window.onload = function() {
    """
    
    # Add JavaScript data and function calls for each heatmap
    for i, sample in enumerate(samples):
        source = sample['source']
        prediction = sample['prediction']
        is_correct = sample['correct']
        attention_weights = np.array(sample['attention_weights'])
        
        # Limit to the length of prediction
        attention_weights = attention_weights[:len(prediction), :len(source)].tolist()
        
        # Add JavaScript to draw this heatmap
        html += f"""
            // Data for heatmap {i+1}
            const source{i+1} = {json.dumps(list(source))};
            const target{i+1} = {json.dumps(list(prediction))};
            const weights{i+1} = {json.dumps(attention_weights)};
            drawHeatmap('heatmap-canvas-{i+1}', source{i+1}, target{i+1}, weights{i+1}, {str(is_correct).lower()});
        """
    
    # Close JavaScript and start HTML body
    html += """
        };
    </script>
</head>
<body>
    <h1>Attention Heatmaps (3×3 Grid)</h1>
    
    <div class="grid-container">
    """
    
    # Add each heatmap container to the HTML
    for i, sample in enumerate(samples):
        source = sample['source']
        prediction = sample['prediction']
        is_correct = sample['correct']
        
        html += f"""
        <div class="heatmap-container">
            <div class="heatmap-title {'correct' if is_correct else 'incorrect'}">
                {'✓' if is_correct else '✗'} {source} → {prediction}
            </div>
            <div class="heatmap-canvas">
                <canvas id="heatmap-canvas-{i+1}" width="250" height="250"></canvas>
            </div>
        </div>
        """
    
    # Add legend and close HTML
    html += """
    </div>
    
    <div class="legend">
        <p>Attention Weight</p>
        <div class="legend-gradient"></div>
        <div class="legend-labels">
            <span>0.0</span>
            <span>0.2</span>
            <span>0.4</span>
            <span>0.6</span>
            <span>0.8</span>
            <span>1.0</span>
        </div>
    </div>
</body>
</html>
    """
    
    # Write the HTML to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Compact HTML attention heatmaps saved to {output_file}")
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate compact HTML attention heatmaps')
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to the JSON file containing predictions with attention weights')
    parser.add_argument('--output_file', type=str, default='compact_heatmaps.html',
                        help='Path to save the HTML file')
    
    args = parser.parse_args()
    
    generate_compact_heatmaps_html(args.predictions_file, args.output_file)
