import argparse
from transformers import (
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch
from preprocess import tokenizer  # Ensure to import your tokenizer

def load_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to("cuda")
    return model

def train_model(train_dataset, test_dataset, tokenizer, num_epochs):
    model = load_model("Salesforce/codet5-small")
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True
    )
    
    # Adding Early Stopping Callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        callbacks=[early_stopping]
    )
    
    trainer.train()
    trainer.save_model("./trained_model")
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with customizable epochs and early stopping.")
    
    # Argument for the number of epochs
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of training epochs")
    
    # Parse the arguments
    args = parser.parse_args()

    # Assume train and test datasets are preloaded (or add code to load them)
    from data_loader import load_code_dataset
    from preprocess import preprocess_dataset

    dataset = load_code_dataset()
    tokenized_datasets = preprocess_dataset(dataset)
    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # Train the model with the specified number of epochs
    trainer = train_model(train_dataset, test_dataset, tokenizer, args.num_epochs)

