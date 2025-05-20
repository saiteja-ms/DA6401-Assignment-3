import json

def compare_models(vanilla_file, attention_file):
    # Load predictions
    with open(vanilla_file, 'r', encoding='utf-8') as f:
        vanilla_preds = json.load(f)
    
    with open(attention_file, 'r', encoding='utf-8') as f:
        attention_preds = json.load(f)
    
    # Convert to dictionaries for easier comparison
    vanilla_dict = {p['source']: p for p in vanilla_preds}
    attention_dict = {p['source']: p for p in attention_preds}
    
    # Calculate accuracies
    vanilla_correct = sum(1 for p in vanilla_preds if p['correct'])
    attention_correct = sum(1 for p in attention_preds if p['correct'])
    total = len(vanilla_preds)
    
    vanilla_acc = vanilla_correct / total if total > 0 else 0
    attention_acc = attention_correct / total if total > 0 else 0
    
    print(f"Vanilla model accuracy: {vanilla_acc:.4f} ({vanilla_correct}/{total})")
    print(f"Attention model accuracy: {attention_acc:.4f} ({attention_correct}/{total})")
    print(f"Improvement: {(attention_acc - vanilla_acc) * 100:.2f}%")
    
    # Find examples where attention corrected vanilla
    corrected_examples = []
    for source, attn_pred in attention_dict.items():
        if source in vanilla_dict:
            vanilla_pred = vanilla_dict[source]
            if not vanilla_pred['correct'] and attn_pred['correct']:
                corrected_examples.append({
                    'source': source,
                    'target': attn_pred['target'],
                    'vanilla_prediction': vanilla_pred['prediction'],
                    'attention_prediction': attn_pred['prediction']
                })
    
    print(f"\nFound {len(corrected_examples)} examples where attention model corrected vanilla model's errors.")
    
    # Print some examples
    print("\nSample corrected examples:")
    for i, example in enumerate(corrected_examples[:10]):  # Show up to 10 examples
        print(f"{i+1}. Source: {example['source']}")
        print(f"   Target: {example['target']}")
        print(f"   Vanilla prediction: {example['vanilla_prediction']} ✗")
        print(f"   Attention prediction: {example['attention_prediction']} ✓")
        print()
    
    return corrected_examples

# Run the comparison
corrected = compare_models(
    'vanilla_best_predictions.json',
    'attention_best_predictions.json'
)
