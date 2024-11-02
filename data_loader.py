# data_loader.py
from datasets import load_dataset

def load_code_dataset():
    # Load a small subset of the CodeSearchNet Python dataset for testing
    dataset = load_dataset("code_search_net", "python", split="train[:1%]")
    return dataset

