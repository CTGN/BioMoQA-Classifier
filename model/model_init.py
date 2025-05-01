
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
from utils import *

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute evaluation metrics from model predictions."""
    logits, labels = eval_pred
    scores = 1 / (1 + np.exp(-logits.squeeze()))  # Sigmoid
    predictions = (scores > 0.5).astype(int)
    f1 = evaluate.load("f1").compute(predictions=predictions, references=labels)
    accuracy = evaluate.load("accuracy").compute(predictions=predictions, references=labels)
    precision = evaluate.load("precision").compute(predictions=predictions, references=labels)
    optimal_threshold = plot_roc_curve(labels, scores, logger=logger, plot_dir=CONFIG["plot_dir"], data_type="val")
    
    return f1 | accuracy | precision | {"optim_threshold": optimal_threshold}



class LossPlottingCallback(TrainerCallback):
    """Callback to log and plot training and validation losses."""
    def __init__(self):
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        logs = state.log_history[-1] if state.log_history else {}
        print("log history :",state.log_history)
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.eval_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.xticks(range(len(self.train_losses)))
        plt.savefig(os.path.join(CONFIG["plot_dir"], "loss_curve.png"))
        plt.close()
        logger.info(f"Loss plot saved to {CONFIG['plot_dir']}/loss_curve.png")


class CustomTrainingArguments(TrainingArguments):
    def __init__(self,loss_type: str,pos_weight: Optional[float] = None,alpha: Optional[float] = None,gamma: Optional[float] = None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.loss_type = loss_type

        if loss_type == "BCE":
            if pos_weight is None:
                # Calculate default pos_weight if not provided
                self.pos_weight = 5.0
                logger.info(f"Positive weight parameter set to default (in TrainingArguments) -> pos_weight={self.pos_weight}")
            else:
                self.pos_weight =pos_weight
                logger.info(f"pos_weight value in TrainingArguments : {self.pos_weight}")
        elif loss_type == "focal":
            self.alpha = alpha if alpha is not None else 0.25  # Default alpha
            self.gamma = gamma if gamma is not None else 2.0  # Default gamma
            logger.info(f"Alpha and gamma values before training (in TrainingArguments): -> alpha={self.alpha} and gamma={self.gamma}")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        


class CustomTrainer(Trainer):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs: bool = False,num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)
        
        if self.args.loss_type == "BCE":
            pos_weight=self.args.pos_weight
            pos_weight=torch.tensor(pos_weight,device=self.model.device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(logits, labels.float())
        elif self.args.loss_type == "focal":
            loss = sigmoid_focal_loss(logits, labels.float(), alpha=self.args.alpha, gamma=self.args.gamma, reduction="mean")
        
        return (loss, outputs) if return_outputs else loss
    
class LearningRateCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.scheduler = None  # Stores the scheduler reference
        
        def on_step_end(self, args, state, control, **kwargs):
            # Capture the scheduler after each step
            self.scheduler = kwargs.get("scheduler", self.scheduler)
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and self.scheduler is not None:
                # Get the latest learning rate from the scheduler
                logs["learning_rate"] = self.scheduler.get_last_lr()[0]
        def on_train_begin(self, args, state, control, **kwargs):
            print("STATE at beggining of training : ",state)
            return super().on_train_begin(args, state, control, **kwargs)
        def on_train_end(self, args, state, control, **kwargs):
            print("STATE at ending of training  : ",state)
            return super().on_train_end(args, state, control, **kwargs)
        
