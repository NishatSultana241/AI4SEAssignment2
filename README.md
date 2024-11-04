# PredictIf(Training a Transformer model for Predicting if statements)
**Built by the following authors**
Nishat Sultana

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

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
You can train the model using the model_train.py script and also can use the already trained model using the main.py script.

### Model Training:
The model can be trained using the preferred number of epochs from the command line argument.
```bash
python train_model.py --num_epochs 5
```
### Generating Responses from the Trained and Saved Model:
To check any particular data from the test set just pass the --index of that particular data.
```bash
python main.py --masked_data masked_data.csv --index 10 --output_csv single-test-result.csv
```
To check the value over the whole test set no need to provide the index value.
```bash
python main.py --masked_data masked_data.csv --output_csv full-test-results.csv
```
### Features:
Write down the features




