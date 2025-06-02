CONFIG = {
    "seed": 42,
    "logging_dir": "./logs",
    "plot_dir": "/home/leandre/Projects/BioMoQA_Playground/plots",
    "final_model_dir": "./final_model",
    "num_labels": 1,
    "default_training_args": {
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "save_total_limit": 1,
        "learning_rate": None,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 10,
        "fp16": False,
        "logging_strategy": "epoch",
        "report_to": "tensorboard",
    },
}
