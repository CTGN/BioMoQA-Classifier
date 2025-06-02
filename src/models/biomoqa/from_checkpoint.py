#TODO : load a given checkpoint and simply evaluate it on the test set
from src.utils.utils import *
from model_init import *
import json

set_random_seeds(CONFIG["seed"])


def from_checkpoint(checkpoint_path):
    # Load and tokenize datasets
    train_ds, val_ds, test_ds = load_datasets()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenized_train, tokenized_val,tokenized_test = tokenize_datasets(train_ds, val_ds,test_ds, tokenizer)
    loss_type="BCE" 

    #TODO : see the doc
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    with open(checkpoint_path + "params.json", 'r') as f:
        best_config = json.load(f)


    logger.info(f"Best config : {best_config}")


    #visualize_ray_tune_results(best_config, logger, plot_dir=CONFIG['plot_dir'])
    #plot_trial_performance(best_config,logger=logger,plot_dir=CONFIG['plot_dir'])

    #TODO : Perform ensemble learning with 10 independent models by looping here, then take the majority vote for each test instance. 
    #TODO :Check how to do ensemble learning with transformers
    model=BertForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=CONFIG["num_labels"])
    
    # Set up training arguments
    training_args = CustomTrainingArguments(
            output_dir=CONFIG["output_dir"],
            seed=CONFIG["seed"],
            data_seed=CONFIG["seed"],
            **CONFIG["default_training_args"],
            loss_type=loss_type,
        )
    
    training_args.pos_weight = best_config["pos_weight"] if loss_type == "BCE" else None
    training_args.alpha = best_config["alpha"] if loss_type == "focal" else None
    training_args.gamma = best_config["gamma" ]if loss_type == "focal" else None
    training_args.learning_rate= best_config["learning_rate"]
    training_args.num_train_epochs=best_config["num_train_epochs"]

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,#! Be careful with this
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()
    eval_results = trainer.evaluate(tokenized_test)
    logger.info(f"Evaluation results on test set: {eval_results}")

    if loss_type=="BCE":
        final_model_path = os.path.join(CONFIG["final_model_dir"], "best_model_cross_val_BCE")
    elif loss_type=="focal":
        final_model_path = os.path.join(CONFIG["final_model_dir"], "best_model_cross_val_Focal")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    trainer.save_model(final_model_path)
    logger.info(f"Best model saved to {final_model_path}")

    print("On test Set (with threshold 0.5): ")
    # Compute detailed metrics
    predictions = trainer.predict(tokenized_test)
    
    scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
    preds = (scores > 0.5).astype(int)
    detailed_metrics(preds, test_ds["labels"])
    plot_roc_curve(test_ds["labels"],scores,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")
    plot_precision_recall_curve(test_ds["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")
    
    with open(checkpoint_path + "result.json", 'r') as f:
            best_results = json.load(f)
    
    threshold = best_results["optim_threshold"]
    print("\nOn test Set (optimal threshold according to cross validation on the training set): ")
    preds = (scores > threshold).astype(int)
    detailed_metrics(preds, test_ds["labels"])
    plot_precision_recall_curve(test_ds["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

    torch.cuda.empty_cache()


    return eval_results

if __name__ == "__main__":
    from_checkpoint("/home/leandre/Projects/BioMoQA_Playground/model/ray_results/train_bert_2025-04-13_01-20-35/train_bert_42262298_15_learning_rate=0.0000,num_train_epochs=2,pos_weight=9.2888,weight_decay=0.1911_2025-04-13_07-29-37/")