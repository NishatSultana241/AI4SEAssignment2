import argparse
import csv
import textwrap
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import torch.nn.functional as F
from Levenshtein import ratio as levenshtein_ratio

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

# Function to generate a prediction and calculate the prediction score
def generate_prediction_with_score(model, tokenizer, masked_function):
    input_ids = tokenizer(masked_function, return_tensors="pt").input_ids.to("cuda")
    output_ids = model.generate(input_ids, max_new_tokens=120, output_scores=True, return_dict_in_generate=True)

    # Decode the generated prediction
    output_tokens = output_ids.sequences[0]
    prediction = tokenizer.decode(output_tokens, skip_special_tokens=True)

    # Calculate prediction score based on softmax probabilities
    scores = output_ids.scores  # List of logits for each generated token
    token_probs = [F.softmax(score, dim=-1) for score in scores]

    # Extract the probability of the selected tokens in the generated sequence
    selected_probs = [
        token_probs[i][0, token_id].item()
        for i, token_id in enumerate(output_tokens[1:])  # Skip initial token in output
    ]

    # Calculate the average probability (confidence) of the sequence
    if selected_probs:
        prediction_score = sum(selected_probs) / len(selected_probs) * 100
    else:
        prediction_score = 0.0  # Handle cases where no probabilities are extracted

    return prediction, prediction_score

# Function to check if the prediction is correct using a similarity threshold
def is_prediction_correct(expected, predicted, threshold=0.7):
    similarity_score = levenshtein_ratio(expected, predicted)
    return similarity_score >= threshold

# Function for pretty printing
def pretty_print(title, content):
    print(f"{title}:")
    print(textwrap.indent(textwrap.fill(content, width=50), prefix="    "))
    print()

# Function to output results to a CSV file
def output_to_csv(output_path, results):
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Input', 'Is Correct', 'Expected If Condition', 'Predicted If Condition', 'Prediction Score'])
        writer.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions using a masked dataset, calculate prediction scores, and output results.")
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
            prediction, prediction_score = generate_prediction_with_score(model, tokenizer, masked_function)

            # Check if the prediction is correct using Levenshtein similarity
            is_correct = is_prediction_correct(target_block.strip(), prediction.strip(), threshold=0.7)

            # Print results
            pretty_print(f"Original Function at Index {args.index}", masked_function)
            pretty_print("Generated Prediction", prediction)
            pretty_print("Expected If Condition", target_block)
            pretty_print("Is Correct", str(is_correct))
            pretty_print("Prediction Score (Confidence)", f"{prediction_score:.2f}")

            # Append results to list
            results.append([masked_function, is_correct, target_block, prediction, f"{prediction_score:.2f}"])
        else:
            print(f"Invalid index. Please provide an index between 0 and {len(masked_data) - 1}.")
    else:
        # Process the entire dataset
        for i, (masked_function, target_block) in enumerate(masked_data):
            prediction, prediction_score = generate_prediction_with_score(model, tokenizer, masked_function)

            # Check if the prediction is correct using Levenshtein similarity
            is_correct = is_prediction_correct(target_block.strip(), prediction.strip(), threshold=0.7)

            # Append results to the list
            results.append([masked_function, is_correct, target_block, prediction, f"{prediction_score:.2f}"])

            # Print progress
            print(f"Processed sample {i + 1}/{len(masked_data)}")

    # Output results to CSV
    output_to_csv(args.output_csv, results)
    print(f"Results written to {args.output_csv}")
