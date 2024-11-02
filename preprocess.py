# preprocess.py
import re
from transformers import AutoTokenizer

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

def mask_if_statements(examples):
    pattern = re.compile(r"if\s+.*:\n(\s+.*\n)*")
    masked_code = [pattern.sub("<extra_id_0>\n", code) for code in examples['func_code_string']]
    return {"input_text": masked_code, "target_text": examples['func_code_string']}

def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=512).input_ids
    model_inputs["labels"] = labels
    return model_inputs

def preprocess_dataset(dataset):
    # Apply masking and tokenization
    masked_dataset = dataset.map(mask_if_statements, batched=True)
    tokenized_datasets = masked_dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

