# evaluate.py
def evaluate_model(trainer):
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

