# PredictIf(Training a Transformer model for Predicting if statements)
**Built by the following authors**
Nishat Sultana

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Pre-processed Data](#Data)

## Installation
Follow these steps to install and set up the project:

### Clone the repository:
```bash
git clone https://github.com/NishatSultana241/PredictIF.git
```
### Navigate to the project directory:
```bash
cd PredictIF
```
### Install dependencies:
```bash
pip install -r requirements.txt
```
### Usage:
To use this fine-tuned transformer model first you have to train it using the train_model.py script and you can train for any number of epochs. After running the training script the trained_model directory will be storing a model that has been trained for  your prefered number of epochs with early stopping.

### Model Training:
The model can be trained using the preferred number of epochs from the command line argument.
```bash
python train_model.py --num_epochs <number of epochs you want to train your model>
python train_model.py --num_epochs 5
```
### Generating Responses from the Trained and Saved Model:
To check any particular data from the test set just pass the index of that particular data from the test set provided-testset.csv.
```bash
python main.py --masked_data <Original masked test set name> --index <Specific entry of test set>  --output_csv <Name of the output file>
python main.py --masked_data masked_data.csv --index 10 --output_csv single-test-result.csv
```
To check the value over the whole test set no need to provide the index value.
```bash
python main.py --masked_data <original Masked Test set Name> --output_csv <Name of the Output File>
python main.py --masked_data masked_data.csv --output_csv full-test-results.csv
```
### Features:
    1. Fine-tuned a Pre-trained T5 Model to predict the masked "If" statements.
    2. Fine-tuned on a large corpus of CodeSearchNet can be easily expanded for other predictions. 
    3. Calculates the prediction score and generates a csv file with the Actual Code masked,Generated Prediction,Expected If Condition etc.

### Data:
All of out pre-processed original,cleaned and masked data could be accessed here in this [link](https://drive.google.com/drive/folders/1fWlQioHs0yqGgH1Epxskn1qhuPTOAt9H?usp=sharing). The masked data is added in the Github repo to train and evaluate the code smoothly.
    




