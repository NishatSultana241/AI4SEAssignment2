import argparse
import csv
import textwrap
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

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

# Function to generate a specific prediction
def generate_specific_prediction(model, tokenizer, masked_function):
    input_ids = tokenizer(masked_function, return_tensors="pt").input_ids.to("cuda")
    output_ids = model.generate(input_ids, max_new_tokens=120)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

# Function for pretty printing
def pretty_print(title, content):
    print(f"{title}:")
    print(textwrap.indent(textwrap.fill(content, width=50), prefix="    "))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a specific prediction using a masked dataset.")
    parser.add_argument('--index', type=int, help="Index of the function in the masked dataset")
    args = parser.parse_args()

    # Path to the masked dataset
    masked_dataset_path = 'masked_data.csv'

    # Load the masked dataset
    masked_data = load_masked_dataset(masked_dataset_path)

    if args.index is not None:
        if 0 <= args.index < len(masked_data):
            masked_function, target_block = masked_data[args.index]
            prediction = generate_specific_prediction(model, tokenizer, masked_function)
            
            pretty_print(f"Original Function at Index {args.index}", masked_function)
            pretty_print("Generated Prediction", prediction)
        else:
            print(f"Invalid index. Please provide an index between 0 and {len(masked_data) - 1}.")
    else:
        print("Please provide an index using the --index argument.")

