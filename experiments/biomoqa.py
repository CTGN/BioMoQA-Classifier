import argparse
from time import perf_counter
import torch
import ray
import logging
import sys 
import os
#TODO : add int args for n_trials...

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Adjust ".." based on your structure

# Add it to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.models.biomoqa import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-om","--opt-metric", help="Metric to use when optimizing the hyperparamters of the model")
    parser.add_argument("-t","--title", help="Use titles as features",action="store_true")
    parser.add_argument("-k","--keywords", help="use keywords as features",action="store_true")
    parser.add_argument("-e","--ensemble", help="Enables ensemble learning",action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    begin_pipeline=perf_counter()
    ray.init(num_gpus=torch.cuda.device_count())
    pipeline=TrainPipeline(None,ensemble=args.ensemble,hpo_metric="eval_f1",with_title=args.title,with_keywords=args.keywords,n_folds=3,n_trials=12,num_runs=2,nb_optional_negs=0,model_names=["dmis-lab/biobert-v1.1", "google-bert/bert-base-uncased"])
    logger.info(f"with_title : {pipeline.with_title}")
    logger.info(f"with_keywords : {pipeline.with_keywords}")
    #pipeline.svm()
    #pipeline.random_forest()
    """
    for model_name in pipeline.model_names:
        pipeline.svm_bert(model_name)
    """
    pipeline.whole_pipeline()
    torch.cuda.empty_cache()  # Clear CUDA cache after pipeline
    end_pipeline=perf_counter()
    pipeline_runtime=end_pipeline-begin_pipeline
    logger.info(f"Total running time : {pipeline_runtime}")
    #TODO : Add a statistical test function

    return None


if __name__ == "__main__":
    main()