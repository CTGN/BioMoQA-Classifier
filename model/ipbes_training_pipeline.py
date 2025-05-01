from optimization_cross_val import *
from utils import *
import sys
import datasets
#TODO :

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Adjust ".." based on your structure

# Add it to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data import data_pipeline

def pipeline(classification_type,ensemble=False):
    """
    This function runs the entire pipeline for training and evaluating a model using cross-validation.
    It includes data loading, preprocessing, model training, and evaluation.
    """
    data_dict=data_pipeline()
    print(data_dict[classification_type])
    train,test,_=data_dict[classification_type]

    def is_label(batch,label):
        batch_bools=[]
        for ex_label in batch['labels']:
            if ex_label == label:
                batch_bools.append(True)
            else:
                batch_bools.append(False)
        return batch_bools

    # Assuming your dataset has a 'label' column (adjust if needed)
    pos_train = train.filter(lambda x : is_label(x,1), batched=True, batch_size=1000, num_proc=os.cpu_count())
    neg_train = train.filter(lambda x : is_label(x,0), batched=True, batch_size=1000, num_proc=os.cpu_count())
    pos_test = test.filter(lambda x : is_label(x,1), batched=True, batch_size=1000, num_proc=os.cpu_count())
    neg_test = test.filter(lambda x : is_label(x,0), batched=True, batch_size=1000, num_proc=os.cpu_count())
    print("Number of train positives",len(pos_train))
    print("Number of train negatives",len(neg_train))
    print("Number of test positives",len(pos_test))
    print("Number of test negatives",len(neg_test))
    num_pos = len(pos_train)

    # Ensure there are more negatives than positives before subsampling
    if len(neg_train) > num_pos:
        neg_subset_train = neg_train.shuffle(seed=42).select(range(10*num_pos))
    else:
        neg_subset_train = neg_train  # Fallback (unlikely in your case)
    
    if len(neg_test) > len(pos_test):
        neg_subset_test = neg_test.shuffle(seed=42).select(range(10*len(pos_test)))
    else:
        neg_subset_test = neg_test

    balanced_train = datasets.concatenate_datasets([pos_train, neg_subset_train])
    balanced_train = balanced_train.shuffle(seed=42)  # Final shuffle
    balanced_test = datasets.concatenate_datasets([pos_test, neg_subset_test])
    balanced_test = balanced_test.shuffle(seed=42)  # Final shuffle

    balanced_train = balanced_train.rename_column("abstract", "text")
    balanced_test = balanced_test.rename_column("abstract", "text")
    
    print(f"Balanced dataset size: {len(balanced_train)}")
    print(f"Test dataset size: {len(balanced_test)}")
    
    if ensemble==False:
        #For ensemble learning : make a function that execute the optimize_model_cross_val function for 5 different model names 
        analysis=optimize_model_cross_val(balanced_train,balanced_test,loss_type="BCE",n_trials=2,n_fold=5)
        print(analysis)
    elif ensemble==True:
        #The following is enough for comaprison purposes, for proper ensemble learning, we should go through each prediction and take the majority vote (see the 2 methods ?)
        model_names = ["model1", "model2", "model3", "model4", "model5"]
        analyses = []
        preds_list = []

        for model_name in model_names:
            analysis,preds = optimize_model_cross_val(balanced_train, balanced_test, loss_type="focal", n_trials=2, n_fold=5, model_name=model_name)
            analyses.append(analysis)
            preds_list.append(preds)
            print(f"Analysis for {model_name}: {analysis}")

        #Then we can take the majority vote
        all_preds = np.vstack(preds_list).T  # shape = (N, 5)

        def majority_vote(arr):
            # arr is of shape (n_models,)
            values, counts = np.unique(arr, return_counts=True)
            return values[np.argmax(counts)]

        # apply to each row (i.e. each sample)
        majority_preds = np.apply_along_axis(majority_vote, axis=1, arr=all_preds)
        return analyses, majority_preds

    #Compare different training methods
    return analysis

if __name__ == "__main__":
    ray.init(num_gpus=torch.cuda.device_count())
    # Run the pipeline for a specific classification type
    classification_type = "SUA"  # Change this to "IAS" or "VA" as needed
    results= pipeline(classification_type)