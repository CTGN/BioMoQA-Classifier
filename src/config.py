CONFIG = {
    "seed": 42,
    "plot_dir": "/home/leandre/Projects/BioMoQA_Playground/plots",
    "final_model_dir": "./final_model",
    "num_labels": 1,
    "default_training_args": {
        "save_total_limit": 1,
        "learning_rate": None,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 10,
        "fp16": False,
        "logging_strategy": "epoch",
        "report_to": "tensorboard",
    },
}
