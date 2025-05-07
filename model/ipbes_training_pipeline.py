from optimization_cross_val import *
from utils import *
import sys
import datasets
from time import perf_counter

#? How do we implement ensemble learning with cross-validation
# -> like i did i guess
# look for other ways ? ask julien ?
# TODO : change ensemble implementation so that it takes scores and returns scores (average ? ) -> see julien's covid paper
# TODO : compare with and twithout title
# TODO : Put longer number of epoch and implement early stopping, ask julien to explain again why tha adma opt implies that early stopping is better for bigger epochs

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Adjust ".." based on your structure

# Add it to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data import data_pipeline

def balance_dataset(dataset):
    """
    - Performs undersampling on the negatives
    - Renames the abstract column -> we should not hqve to do that 
    """

     
    def is_label(batch,label):
        batch_bools=[]
        for ex_label in batch['labels']:
            if ex_label == label:
                batch_bools.append(True)
            else:
                batch_bools.append(False)
        return batch_bools

    # Assuming your dataset has a 'label' column (adjust if needed)
    pos = dataset.filter(lambda x : is_label(x,1), batched=True, batch_size=1000, num_proc=os.cpu_count())
    neg = dataset.filter(lambda x : is_label(x,0), batched=True, batch_size=1000, num_proc=os.cpu_count())
    logger.info(f"Number of positives: {len(pos)}")
    logger.info(f"Number of negatives: {len(neg)}")
    num_pos = len(pos)

    # Ensure there are more negatives than positives before subsampling
    if len(neg) > num_pos:
        #TODO : Change the proportion value her for les or more imbalance -> compare different values, plot ? try less
        neg_subset_train = neg.shuffle(seed=42).select(range(num_pos))
    else:
        neg_subset_train = neg  # Fallback (unlikely in your case)

    balanced_ds = datasets.concatenate_datasets([pos, neg_subset_train])
    balanced_ds = balanced_ds.shuffle(seed=42)  # Final shuffle

    balanced_ds = balanced_ds.rename_column("abstract", "text")
    logger.info(f"Balanced columns: {balanced_ds.column_names}")
    logger.info(f"Balanced dataset size: {len(balanced_ds)}")

    return balanced_ds

#TODO : compare the loss functions inside the pipeline function to have the same test set for each run
def pipeline(classification_type,loss_type="BCE",ensemble=False,with_title=False,n_fold=5):
    """
    This function runs the entire pipeline for training and evaluating a model using cross-validation.
    It includes data loading, preprocessing, model training, and evaluation.
    """
    logger.info(f"Running pipeline for {classification_type} with loss type {loss_type} and ensemble={ensemble}")

    set_random_seeds(CONFIG["seed"])

    data_dict=data_pipeline()

    logger.info(f"Data for {classification_type}: {data_dict[classification_type]}")
    dataset=data_dict[classification_type]

    # ? When and what should we balance ? 
    # the when depends on what. If only training is balances then it should be done inside the optimization_cros_val function
    #TODO : even if done on the whole dataset, we can move it into optimization_cros_val
    logger.info(f"Balancing dataset...")
    balanced_dataset = balance_dataset(dataset)

    #First we do k-fold cross validation for testing
    #TODO : ensure that this does not change when running this pipeline different times for comparisons purposes
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=CONFIG["seed"])

    if with_title:
        folds = list(skf.split(balanced_dataset[['text','title']], balanced_dataset['labels']))

    elif not with_title:
        folds = list(skf.split(balanced_dataset['text'], balanced_dataset['labels']))

    else:
        raise ValueError("Invalid value for with_title. It must be True or False.")
    
    if ensemble==False:
        #For ensemble learning : make a function that execute the optimize_model_cross_val function for 5 different model names 
        avg_metrics,test_metrics, scores_by_fold =optimize_model_cross_val(balanced_dataset, folds, loss_type=loss_type, n_trials=12)
        logger.info(f"Results: {results}")
        
        return test_metrics,None

    elif ensemble==True:
        logger.info(f"Ensemble learning pipeline")
        
        model_names = ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext","FacebookAI/roberta-base", "dmis-lab/biobert-v1.1", "google-bert/bert-base-uncased"]
        scores_by_model = []

        for i,model_name in enumerate(model_names):
            logger.info(f"Training model {i+1}/{len(model_names)}: {model_name}")
            avg_metrics,test_metrics, scores_by_fold = optimize_model_cross_val(balanced_dataset, folds, loss_type=loss_type, n_trials=3, model_name=model_name, data_type=classification_type)
            torch.cuda.empty_cache()
            scores_by_model.append(scores_by_fold)
            logger.info(f"Metrics for {model_name}: {test_metrics}")

        #Then we can take the majority vote
        all_scores = np.vstack(scores_by_model).T  # shape = (N, 5)
        mean_scores=np.mean(all_scores,axis=1)

        #TODO : take a decision based on this
        #TODO : Implement ensemble metrics -> use it in the whole_pipeline function
        #TODO : retrun ensemble metric or add it to the test metrics (which should be renamed in that case)
        return avg_metrics,test_metrics,mean_scores

    #Compare different training methods
    return None

def whole_pipeline():
    output=[]
    classification_types= ["SUA", "IAS", "VA"]
    for classification_type in classification_types:
        logger.info(f"Running pipeline for {classification_type}")
        avg_metrics,test_metrics,mean_scores=pipeline(classification_type, ensemble=True,n_fold=5)
        #! The use of avg_metrics is wrong here
        output.append(avg_metrics)
        logger.info(f"Metrics for {classification_type}: {avg_metrics}")
    return output

if __name__ == "__main__":
    begin_pipeline=perf_counter()
    ray.init(num_gpus=torch.cuda.device_count())
    final_output=whole_pipeline()
    end_pipeline=perf_counter()
    logger.info(f"Final output: {final_output}")
    pipeline_runtime=end_pipeline-begin_pipeline
    logger.info(f"Total running time : {pipeline_runtime}")
    #TODO : Add a statistical test function