import json
import random
import os

def create_creative_predictions_grid(predictions_file, output_file='vanilla_predictions_grid.html', num_samples=12):
    """
    Creates a creative HTML grid visualization of sample predictions from the vanilla model.
    
    Args:
        predictions_file: Path to the JSON file containing predictions
        output_file: Path to save the HTML file
        num_samples: Number of samples to display
    """
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Separate correct and incorrect predictions
    correct_preds = [p for p in predictions if p['correct']]
    incorrect_preds = [p for p in predictions if not p['correct']]
    
    # Sample predictions for the grid
    num_correct = min(num_samples // 2, len(correct_preds))
    num_incorrect = min(num_samples - num_correct, len(incorrect_preds))
    
    sampled_correct = random.sample(correct_preds, num_correct) if correct_preds else []
    sampled_incorrect = random.sample(incorrect_preds, num_incorrect) if incorrect_preds else []
    
    samples = sampled_correct + sampled_incorrect
    random.shuffle(samples)
    
    # Start building the HTML content
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Vanilla Model Predictions</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #333;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-gap: 20px;
            max-width: 1000px;
            margin: 0 auto;
        }
        .prediction-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .prediction-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .correct {
            border-left: 5px solid #4CAF50;
        }
        .incorrect {
            border-left: 5px solid #F44336;
        }
        .status-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        .correct-badge {
            background-color: #4CAF50;
        }
        .incorrect-badge {
            background-color: #F44336;
        }
        .source {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .target-container {
            display: flex;
            margin-bottom: 10px;
            gap: 10px;
        }
        .label {
            font-weight: bold;
            width: 100px;
            color: #666;
            flex-shrink: 0;
        }
        .target, .prediction {
            flex-grow: 1;
            font-size: 16px;
            word-break: break-word;
            padding-left: 5px;
        }
        .prediction-container {
            display: flex;
            margin-bottom: 5px;
        }
        .attestation {
            font-size: 12px;
            color: #666;
            text-align: right;
            margin-top: 10px;
        }
        .highlight {
            background-color: #FFECB3;
            padding: 2px;
            border-radius: 3px;
        }
        .error {
            color: #F44336;
            text-decoration: underline;
            text-decoration-style: wavy;
            text-decoration-color: #F44336;
        }
    </style>
</head>
<body>
    <h1>Vanilla Model Predictions</h1>
    
    <div class="grid-container">
    """
    
    # Add each prediction card to the HTML
    for sample in samples:
        source = sample['source']
        target = sample['target']
        prediction = sample['prediction']
        is_correct = sample['correct']
        attestation = sample.get('attestation', 1)
        
        # Highlight differences between target and prediction
        highlighted_prediction = ""
        if not is_correct:
            for i, char in enumerate(prediction):
                if i < len(target) and char == target[i]:
                    highlighted_prediction += f"{char}"
                else:
                    highlighted_prediction += f"<span class='error'>{char}</span>"
        else:
            highlighted_prediction = prediction
        
        html += f"""
        <div class="prediction-card {'correct' if is_correct else 'incorrect'}">
            <div class="status-badge {'correct-badge' if is_correct else 'incorrect-badge'}">
                {'✓ Correct' if is_correct else '✗ Incorrect'}
            </div>
            <div class="source">{source}</div>
            <div class="target-container">
                <div class="label">Target:</div>
                <div class="target">{target}</div>
            </div>
            <div class="prediction-container">
                <div class="label">Prediction:</div>
                <div class="prediction">{'<span class="highlight">' + prediction + '</span>' if is_correct else highlighted_prediction}</div>
            </div>
            <div class="attestation">Attestation: {attestation}</div>
        </div>
        """
    
    # Close HTML
    html += """
    </div>
</body>
</html>
    """
    
    # Write the HTML to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Creative predictions grid saved to {output_file}")
    return output_file

def copy_predictions_to_github_folder(predictions_file, github_folder='predictions_vanilla'):
    """
    Copies the predictions file to the GitHub folder.
    
    Args:
        predictions_file: Path to the predictions JSON file
        github_folder: Path to the GitHub folder
    """
    # Create the folder if it doesn't exist
    os.makedirs(github_folder, exist_ok=True)
    
    # Copy the file
    import shutil
    destination = os.path.join(github_folder, os.path.basename(predictions_file))
    shutil.copy2(predictions_file, destination)
    
    print(f"Copied predictions to {destination}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create creative predictions grid')
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to the JSON file containing predictions')
    parser.add_argument('--output_file', type=str, default='vanilla_predictions_grid.html',
                        help='Path to save the HTML file')
    args = parser.parse_args()
    
    # Create the creative grid
    create_creative_predictions_grid(args.predictions_file, args.output_file)
    