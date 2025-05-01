from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, DataCollatorWithPadding, TrainingArguments, TrainerCallback, set_seed
import random
from evaluate import load
import numpy as np
import torch
import os
from torchvision.ops import sigmoid_focal_loss
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

import evaluate
evaluate.logging.disable_progress_bar()  
ray.init(
    runtime_env={
        "py_modules": [evaluate],  # Explicitly include evaluate
        "excludes": ["*.csv", "data/*", "results/*"]
    }
)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(SEED)

# Load datasets
train_ds = load_dataset("csv", data_files="data/train/train.csv", split="train")
val_ds = load_dataset("csv", data_files="data/val/val.csv", split="train")
test_ds = load_dataset("csv", data_files="data/test/test.csv", split="train")

# Tokenization setup
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenization(ex):
    return tokenizer(ex["text"], truncation=True)

tokenized_train = train_ds.map(tokenization, batched=True)
tokenized_val = val_ds.map(tokenization, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics setup
f1_metric = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    scores = 1 / (1 + np.exp(-logits.squeeze())) 
    predictions = (scores > 0.5).astype(int)
    return f1_metric.compute(predictions=predictions, references=labels)

# Custom Trainer with focal loss parameters
class CustomTrainer(Trainer):
    def __init__(self, alpha=0.92, gamma=2.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = self.train_dataset["labels"]
        nb_neg = sum(1 for label in labels if label == 0)
        nb_pos = sum(1 for label in labels if label == 1)
        self.pos_weight = torch.tensor(nb_pos / nb_neg, device=self.model.device)
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        loss = sigmoid_focal_loss(logits, labels.float(), 
                                alpha=self.alpha, 
                                gamma=self.gamma, 
                                reduction='mean')
        return (loss, outputs) if return_outputs else loss

# Ray Tune training function
def train_model(config):
    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

    trial_dir = os.path.abspath(os.path.join("./ray_results", tune.get_trial_id()))
    os.makedirs(trial_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=trial_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        fp16=False,
        seed=SEED,
        data_seed=SEED,
        logging_dir=os.path.join(trial_dir, "logs"),
        load_best_model_at_end=True,
        gradient_accumulation_steps=8,
        logging_strategy="epoch",
        report_to="none",
    )

    # Initialize model and trainer
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        alpha=config["alpha"],
        gamma=config["gamma"],
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Train and evaluate
    trainer.train()
    eval_result = trainer.evaluate()
    tune.report(f1=eval_result["eval_f1"])

# Configure and run the hyperparameter search
search_space = {
    "alpha": tune.uniform(0.1, 1.0),
    "gamma": tune.uniform(0.0, 5.0)
}

analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=10,
    resources_per_trial={"cpu": 2, "gpu": 1},
    metric="f1",
    mode="max",
    storage_path="/home/leandre/Projects/BioMoQA_Playground/ray_results",
)

# Get best hyperparameters
best_config = analysis.get_best_config(metric="f1", mode="max")
print(f"Best hyperparameters: {best_config}")

# Train final model with best config
final_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
final_trainer = CustomTrainer(
    model=final_model,
    args=TrainingArguments(
        output_dir="./best_model",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=SEED
    ),
    alpha=best_config["alpha"],
    gamma=best_config["gamma"],
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

final_trainer.train()
final_trainer.save_model("./best_model")

# Evaluate on test set
test_results = final_trainer.evaluate(test_ds)
print(f"Final test results: {test_results}")