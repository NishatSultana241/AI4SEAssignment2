import argparse
import csv
import textwrap
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load pre-trained model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./trained_model").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

# Function to read the masked dataset
def load_masked_dataset(masked_dataset_path):
    masked_data = []
    with open(masked_dataset_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            masked_data.append(row)
    return masked_data

# Function to generate a prediction for a given input
def generate_prediction(model, tokenizer, input_code):
    input_ids = tokenizer(input_code, return_tensors="pt").input_ids.to("cuda")
    output_ids = model.generate(input_ids, max_new_tokens=120)
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return prediction

# Function to calculate BLEU score with smoothing for short sequences
def calculate_bleu_score(expected, predicted):
    expected_tokens = expected.split()
    predicted_tokens = predicted.split()
    smooth_fn = SmoothingFunction().method1  # Apply smoothing to handle short sequences
    score = sentence_bleu([expected_tokens], predicted_tokens, smoothing_function=smooth_fn) * 100  # Normalize to a percentage
    return score

# Function for pretty printing
def pretty_print(title, content):
    print(f"{title}:")
    print(textwrap.indent(textwrap.fill(content, width=50), prefix="    "))
    print()

# Function to output results to a CSV file
def output_to_csv(output_path, results):
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Input', 'Is Correct', 'Expected If Condition', 'Predicted If Condition', 'BLEU Score'])
        writer.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions using a masked dataset, calculate BLEU score, and output results.")
    parser.add_argument('--masked_data', type=str, help="Path to the input masked dataset CSV file", required=True)
    parser.add_argument('--output_csv', type=str, help="Path to the output CSV file", default='output-results.csv')
    parser.add_argument('--index', type=int, help="Index of the function in the masked dataset (optional)")
    args = parser.parse_args()

    # Load the masked dataset
    masked_data = load_masked_dataset(args.masked_data)

    results = []

    if args.index is not None:
        # Process a single entry
        if 0 <= args.index < len(masked_data):
            masked_function, target_block = masked_data[args.index]
            prediction = generate_prediction(model, tokenizer, masked_function)

            # Check if the prediction is correct (exact match)
            is_correct = (prediction.strip() == target_block.strip())

            # Calculate BLEU score
            bleu_score = calculate_bleu_score(target_block.strip(), prediction.strip())

            # Print results
            pretty_print(f"Original Function at Index {args.index}", masked_function)
            pretty_print("Generated Prediction", prediction)
            pretty_print("Expected If Condition", target_block)
            pretty_print("Is Correct", str(is_correct))
            pretty_print("BLEU Score", f"{bleu_score:.2f}")

            # Append results to list
            results.append([masked_function, is_correct, target_block, prediction, f"{bleu_score:.2f}"])
        else:
            print(f"Invalid index. Please provide an index between 0 and {len(masked_data) - 1}.")
    else:
        # Process the entire dataset
        for i, (masked_function, target_block) in enumerate(masked_data):
            prediction = generate_prediction(model, tokenizer, masked_function)

            # Check if the prediction is correct (exact match)
            is_correct = (prediction.strip() == target_block.strip())

            # Calculate BLEU score
            bleu_score = calculate_bleu_score(target_block.strip(), prediction.strip())

            # Append results to list
            results.append([masked_function, is_correct, target_block, prediction, f"{bleu_score:.2f}"])

            # Print progress
            print(f"Processed sample {i + 1}/{len(masked_data)}")

    # Output results to CSV
    output_to_csv(args.output_csv, results)
    print(f"Results written to {args.output_csv}")
