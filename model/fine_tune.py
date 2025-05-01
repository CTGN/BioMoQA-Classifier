import logging
import os
from typing import Dict, List, Tuple, Optional

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.search.hyperopt import HyperOptSearch
import ray
from ray.tune.schedulers import ASHAScheduler
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from torchvision.ops import sigmoid_focal_loss
from optimization import optimize_model
from model_init import *
from utils import *

def fine_tune(
    seed_set: bool = True, loss_type: str = "BCE", model_name: str = CONFIG["model_name"], optimize=False, n_trials:Optional[int] =10
) -> Dict:
    """
    This function create a fine-tuned model from given a hugging face BERT model.
    Its hyperparameters can be optimized ()or not (in which case the hyperparameters are arbitrary).

    Arguments :
        seed_set : if True, set a fixed seed for training and hyperparameters optimization for reproducibility (The seed value is set in CONFIG)
        loss_type : specifies the type of loss function we use (can be either weighted binary cross entropy or focal loss)
        model_name : name of the pre-trained model use for fine-tuning, must be a BERT model from the hugging face library
        optimize : specifies whether the function should optmize the hyperparameters or not. False by default.
        n_trials : if optimize is true, specifies the number of optimization trials 
    
    return :
        eval_results : A dictionary containing results of the final evaluation given by compute_metrics()
    """

    #Sets the seed for reproducibility if asked
    if seed_set:
        set_random_seeds(CONFIG["seed"])

    # Load and tokenize datasets
    train_ds, val_ds, test_ds = load_datasets()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_train, tokenized_val,tokenized_test = tokenize_datasets(train_ds, val_ds,test_ds, tokenizer)

    #TODO : see the doc
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #We initialize the training arguments of the model
    training_args = CustomTrainingArguments(
            output_dir=CONFIG["output_dir"],
            seed=CONFIG["seed"],
            data_seed=CONFIG["seed"],
            **CONFIG["default_training_args"],
            loss_type=loss_type,
        )
    
    #Hyperparameters optimization process
    if optimize:
        best_run=optimize_model(loss_type=loss_type,model_name=model_name,seed_set=seed_set,n_trials=n_trials)
        best_hyperparameters = best_run.hyperparameters
        training_args.pos_weight = best_hyperparameters.get("pos_weight") if loss_type == "BCE" else None
        training_args.alpha = best_hyperparameters.get("alpha") if loss_type == "focal" else None
        training_args.gamma = best_hyperparameters.get("gamma") if loss_type == "focal" else None
        training_args.learning_rate=best_hyperparameters.get("learning_rate")
    else:
        training_args.pos_weight = 8.31156 if loss_type == "BCE" else None
        training_args.alpha = 0.9 if loss_type == "focal" else None
        training_args.gamma = 2 if loss_type == "focal" else None
    
    

    print("LEARNING RATE : ", training_args.learning_rate)
    trainer = CustomTrainer(
        model=BertForSequenceClassification.from_pretrained(model_name, num_labels=CONFIG["num_labels"]),
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    final_model_path = os.path.join(CONFIG["final_model_dir"], "best_model")
    trainer.save_model(final_model_path)
    logger.info(f"Best model saved to {final_model_path}")

    print("On validation Set : ")
    # Compute detailed metrics
    predictions = trainer.predict(tokenized_val)
    scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
    preds = (scores > 0.5).astype(int)
    detailed_metrics(preds, val_ds["labels"])

    plot_roc_curve(val_ds["labels"],scores,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="val")
    plot_precision_recall_curve(val_ds["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="val")

    print("On test Set : ")
    # Compute detailed metrics
    predictions = trainer.predict(tokenized_test)
    scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
    preds = (scores > 0.5).astype(int)
    detailed_metrics(preds, test_ds["labels"])
    plot_roc_curve(test_ds["labels"],scores,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")
    plot_precision_recall_curve(test_ds["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

    torch.cuda.empty_cache()

    return eval_results

if __name__ == "__main__":
    #results=fine_tune(loss_type='focal',optimize=True,n_trials=5)
    #print(results)

    results=fine_tune(loss_type='focal',optimize=True,n_trials=100)
    print(results)
