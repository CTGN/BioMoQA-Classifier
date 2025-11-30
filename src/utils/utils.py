import os
import numpy as np
from ray import tune
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
import random
import gc
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset, load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,multilabel_confusion_matrix
import evaluate
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
from sklearn.metrics import average_precision_score,matthews_corrcoef,ndcg_score,cohen_kappa_score,roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
import sys
from pathlib import Path

from src.config import get_config
from src.utils.plot_style import (
    FIGURE_SIZES, PRIMARY_COLORS, PLOT_PARAMS,
    create_figure, format_axis, save_figure, style_roc_curve, style_pr_curve
)


logger = logging.getLogger(__name__)

def map_name(model_name):
    if model_name == "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract":
        return "BiomedBERT-abs"
    elif model_name == "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext":
        return "BiomedBERT-abs-ft"
    elif model_name == "FacebookAI/roberta-base":
        return "roberta-base"
    elif model_name == "dmis-lab/biobert-v1.1":
        return "biobert-v1"
    elif model_name == "google-bert/bert-base-uncased":
        return "bert-base"
    else:
        return model_name

def save_dataframe(metric_df, path=None, file_name="binary_metrics.csv"):
    """Save dataframe to CSV with proper path resolution"""
    if metric_df is not None:
        if path is None:
            config = get_config()
            path = config.get_path("results", "metrics_dir")
        
        # Ensure path is a Path object
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        full_path = path / file_name
        metric_df.to_csv(full_path, index=False)
        logger.info(f"Metrics stored successfully at {full_path}")
    else:
        raise ValueError("result_metrics is None. Consider running the model before storing metrics.")

def detailed_metrics(predictions: np.ndarray, labels: np.ndarray,scores =None) -> Dict[str, float]:
    """Compute and display detailed metrics including confusion matrix."""
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    fig, ax = create_figure(figsize='medium')
    disp.plot(ax=ax, cmap='Blues')

    config = get_config()
    plot_dir = config.get("plots_dir")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, Path(plot_dir) / "confusion_matrix.png")
    plt.close(fig)

    metrics = {
        **(evaluate.load("f1").compute(predictions=predictions, references=labels) or {}),
        **(evaluate.load("recall").compute(predictions=predictions, references=labels) or {}),
        **(evaluate.load("precision").compute(predictions=predictions, references=labels) or {}),
        **(evaluate.load("accuracy").compute(predictions=predictions, references=labels) or {}),
        "roc_auc" : roc_auc_score(labels,scores) if scores is not None else {},
        "AP":average_precision_score(labels,scores,average="weighted") if scores is not None else {},
        "MCC":matthews_corrcoef(labels,predictions),
        "NDCG":ndcg_score(np.asarray(labels).reshape(1, -1),scores.reshape(1, -1)) if scores is not None else {},
        "kappa":cohen_kappa_score(labels,predictions),
        'TN':tn, 'FP':fp, 'FN':fn, "TP":tp
    }
    
    logger.info(f"Metrics: {metrics}")
    return metrics

def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across libraries."""
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seeds set to {seed}")

def load_datasets(processed: bool = True) -> Tuple[Dataset, Dataset]:
    """Load training, validation, and test datasets from CSV files."""
    train_path = CONFIG["data_paths"]["processed_train"] if processed else CONFIG["data_paths"]["raw_train"]
    train_ds = load_dataset("csv", data_files=train_path, split="train")
    test_ds = load_dataset("csv", data_files=CONFIG["data_paths"]["test"], split="train")
    logger.info(f"Loaded datasets. Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    return train_ds, test_ds

#TODO : check python doc to see what are "*" and "**"
def tokenize_datasets(
    *datasets: Dataset, tokenizer, with_title: bool, with_keywords=False
) -> Tuple[Dataset, ...]:
    """Tokenize one, two, or three datasets."""
    def tokenization(batch: Dict) -> Dict:
        if with_title:
            if with_keywords:
                sep_tok = tokenizer.sep_token or "[SEP]"
                combined = [t + sep_tok + k
                            for t, k in zip(batch["title"], batch["Keywords"])]

                return tokenizer(
                    combined,
                    batch["abstract"],  
                    truncation=True,
                    return_attention_mask=True,
                )
            else:
                return tokenizer(batch["title"], batch["abstract"], truncation=True, max_length=512)
        else:
            if with_keywords:
                return tokenizer(batch["abstract"], batch["Keywords"], truncation=True, max_length=512)
            else:
                return tokenizer(batch["abstract"], truncation=True, max_length=512)

    tokenized_datasets = tuple(
        ds.map(tokenization, batched=True, batch_size=1000, num_proc=os.cpu_count()) for ds in datasets
    )
    logger.info(f"{len(datasets)} datasets tokenized successfully")
    return tokenized_datasets

def plot_roc_curve(y_true, y_scores, logger, plot_dir, data_type=None, metric="eval_f1",store_plot=True):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    metric_scores = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        try:
            if metric == "f1":
                score = f1_score(y_true, y_pred)
            elif metric == "accuracy":
                score = accuracy_score(y_true, y_pred)
            elif metric == "precision":
                score = precision_score(y_true, y_pred)
            elif metric == "recall":
                score = recall_score(y_true, y_pred)
            elif metric == "kappa":
                score = cohen_kappa_score(y_true, y_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        except ValueError:
            score = 0  # Handle edge cases like all one class in y_pred
        metric_scores.append(score)

    optimal_idx = np.argmax(metric_scores)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax1 = create_figure(figsize='medium')
    ax1.plot(fpr, tpr, color=PRIMARY_COLORS['orange'], lw=PLOT_PARAMS['linewidth'],
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color=PRIMARY_COLORS['blue'], lw=1.5, linestyle='--', alpha=0.6)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])

    format_axis(ax1,
                xlabel='False Positive Rate',
                ylabel='True Positive Rate',
                title='Receiver Operating Characteristic',
                grid=True,
                grid_axis='both',
                legend=True,
                legend_kwargs={'loc': 'lower right'})

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(fpr[::10])
    ax2.set_xticklabels([f'{t:.2f}' for t in thresholds[::10]], rotation=45, fontsize=8)
    ax2.set_xlabel('Thresholds')

    filename = f"roc_curve_{data_type}.png" if data_type else "roc_curve.png"
    if store_plot:
        save_figure(fig, os.path.join(plot_dir, filename))
        plt.close(fig)
        logger.info(f"ROC curve saved to {os.path.join(plot_dir, filename)}")
    else:
        plt.show()
        logger.info("ROC curve displayed")

    return optimal_threshold

def plot_precision_recall_curve(y_true, y_scores,logger,plot_dir,data_type=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    fig, ax = create_figure(figsize='medium')
    ax.plot(recall, precision, color=PRIMARY_COLORS['blue'], lw=PLOT_PARAMS['linewidth'],
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')

    format_axis(ax,
                xlabel='Recall',
                ylabel='Precision',
                title='Precision-Recall Curve',
                grid=True,
                grid_axis='both',
                legend=True,
                legend_kwargs={'loc': 'lower left'})

    filename = f"precision_recall_curve{data_type}.png" if data_type else "precision_recall_curve.png"
    save_figure(fig, os.path.join(plot_dir, filename))
    plt.close(fig)
    logger.info(f"Precision-Recall curve saved to {plot_dir}/{filename}")
    return avg_precision

def visualize_ray_tune_results(analysis, logger, plot_dir=None, metric="eval_recall", mode="max"):
    """
    Create visualizations of Ray Tune hyperparameter search results.
    
    Args:
        analysis: Ray Tune analysis object
        logger: Logger instance
        plot_dir: Directory for plots (if None, uses config)
        metric: Metric to optimize (default: "eval_recall")
        mode: Optimization mode ("max" or "min")
    """
    
    if plot_dir is None:
        config = get_config()
        plot_dir = config.get("plots_dir")
    
    # Ensure plot_dir is a Path object
    plot_dir = Path(plot_dir)
    
    # Load experiment data
    df = analysis.dataframe()
    
    # Get best configuration
    best_trial = analysis.get_best_trial(metric=metric, mode=mode)
    best_config = best_trial.config
    
    # Create plots directory
    hyperparams_dir = plot_dir / "hyperparams"
    hyperparams_dir.mkdir(parents=True, exist_ok=True)
    
    # For BCE loss (pos_weight parameter)
    if "pos_weight" in df.columns:
        fig, ax = create_figure(figsize='medium')
        ax.scatter(df["pos_weight"], df[metric], alpha=PLOT_PARAMS['alpha'],
                   color=PRIMARY_COLORS['blue'], s=50)
        if "pos_weight" in best_config:
            ax.axvline(x=best_config["pos_weight"], color=PRIMARY_COLORS['red'],
                       linestyle='--', linewidth=PLOT_PARAMS['linewidth'],
                       label=f"Best pos_weight: {best_config['pos_weight']:.2f}")
        format_axis(ax,
                    xlabel="pos_weight",
                    ylabel=f"{metric}",
                    title=f"Effect of pos_weight on {metric}",
                    grid=True,
                    grid_axis='both',
                    legend=True)
        save_figure(fig, hyperparams_dir / "pos_weight_effect.png")
        plt.close(fig)
        logger.info(f"Pos weight effect plot saved")
    
    # For focal loss (alpha and gamma parameters)
    if "alpha" in df.columns and "gamma" in df.columns:
        # 2D scatter plot with colorbar for metric
        fig, ax = create_figure(figsize='large')
        scatter = ax.scatter(df["alpha"], df["gamma"], c=df[metric],
                             cmap="viridis", s=100, alpha=PLOT_PARAMS['alpha'])

        if "alpha" in best_config and "gamma" in best_config:
            ax.scatter([best_config["alpha"]], [best_config["gamma"]],
                      color=PRIMARY_COLORS['red'], s=200, marker='*',
                      label=f"Best (α={best_config['alpha']:.2f}, γ={best_config['gamma']:.2f})")

        plt.colorbar(scatter, label=metric, ax=ax)
        format_axis(ax,
                    xlabel="Alpha (α)",
                    ylabel="Gamma (γ)",
                    title=f"Effect of Focal Loss Parameters on {metric}",
                    grid=True,
                    grid_axis='both',
                    legend=True)
        save_figure(fig, hyperparams_dir / "focal_params_effect.png")
        plt.close(fig)
        logger.info(f"Focal loss parameters effect plot saved")
        
        # 3D surface plot for better visualization
        from mpl_toolkits.mplot3d import Axes3D

        # Create a grid for the surface plot
        unique_alphas = sorted(df["alpha"].unique())
        unique_gammas = sorted(df["gamma"].unique())

        if len(unique_alphas) > 1 and len(unique_gammas) > 1:
            # Only create surface plot if we have multiple values for both parameters
            X, Y = np.meshgrid(unique_alphas, unique_gammas)
            Z = np.zeros((len(unique_gammas), len(unique_alphas)))

            # Fill the grid with metric values
            for i, gamma in enumerate(unique_gammas):
                for j, alpha in enumerate(unique_alphas):
                    subset = df[(df["alpha"] == alpha) & (df["gamma"] == gamma)]
                    if not subset.empty:
                        Z[i, j] = subset[metric].mean()

            # Create 3D surface plot
            fig = plt.figure(figsize=FIGURE_SIZES['large'])
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

            # Mark the best point
            if "alpha" in best_config and "gamma" in best_config:
                best_alpha_idx = unique_alphas.index(best_config["alpha"]) if best_config["alpha"] in unique_alphas else 0
                best_gamma_idx = unique_gammas.index(best_config["gamma"]) if best_config["gamma"] in unique_gammas else 0
                best_z = Z[best_gamma_idx, best_alpha_idx]
                ax.scatter([best_config["alpha"]], [best_config["gamma"]], [best_z],
                          color=PRIMARY_COLORS['red'], s=200, marker='*')

            ax.set_xlabel("Alpha (α)")
            ax.set_ylabel("Gamma (γ)")
            ax.set_zlabel(metric)
            ax.set_title(f"3D Surface of Focal Loss Parameters vs {metric}", fontweight='bold', pad=20)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=metric)
            save_figure(fig, hyperparams_dir / "focal_params_surface.png")
            plt.close(fig)
            logger.info(f"3D surface plot for focal loss parameters saved")
    
    # Plot training curves for the best trial if available
    if "training_iteration" in df.columns and metric in df.columns:
        best_trial_df = df[df["trial_id"] == best_trial.trial_id]
        if len(best_trial_df) > 1:  # Only plot if we have multiple iterations
            fig, ax = create_figure(figsize='medium')
            ax.plot(best_trial_df["training_iteration"], best_trial_df[metric],
                    marker='o', linestyle='-', linewidth=PLOT_PARAMS['linewidth'],
                    color=PRIMARY_COLORS['blue'])
            format_axis(ax,
                        xlabel="Training Iteration",
                        ylabel=metric,
                        title=f"{metric} Progress for Best Trial",
                        grid=True,
                        grid_axis='both',
                        legend=False)
            save_figure(fig, hyperparams_dir / "best_trial_progress.png")
            plt.close(fig)
            logger.info(f"Best trial progress plot saved")
    
    # Plot parallel coordinates for all parameters
    try:
        from ray.tune.analysis.experiment_analysis import ExperimentAnalysis

        if isinstance(analysis, ExperimentAnalysis):
            ax = None
            try:
                import pandas as pd
                from pandas.plotting import parallel_coordinates

                # Convert analysis dataframe to a format suitable for parallel coordinates
                df = analysis.dataframe()
                if not df.empty:
                    param_columns = [col for col in df.columns if col.startswith("config/")]
                    if param_columns:
                        df_params = df[param_columns + [metric]].dropna()
                        df_params = df_params.rename(columns=lambda x: x.replace("config/", ""))
                        df_params["trial_id"] = df["trial_id"]
                        
                        # Normalize metric for better visualization
                        df_params[metric] = (df_params[metric] - df_params[metric].min()) / (df_params[metric].max() - df_params[metric].min())
                        
                        plt.figure(figsize=(12, 6))
                        parallel_coordinates(df_params, class_column="trial_id", colormap=plt.cm.viridis)
                        plt.title(f"Parallel Coordinates Plot for {metric}")
                        plt.xlabel("Parameters")
                        plt.ylabel("Normalized Metric")
                        plt.grid(True)
                        plt.savefig(hyperparams_dir / "parallel_coordinates.png")
                        plt.close()
                        logger.info(f"Parallel coordinates plot saved")
                    else:
                        logger.warning("No parameter columns found for parallel coordinates plot")
                else:
                    logger.warning("Analysis dataframe is empty, cannot create parallel coordinates plot")
                if ax:
                    plt.title(f"Parallel Coordinates Plot for {metric}")
                    plt.savefig(hyperparams_dir / "parallel_coordinates.png")
                    plt.close()
                    logger.info(f"Parallel coordinates plot saved")
            except Exception as e:
                logger.warning(f"Could not create parallel coordinates plot: {e}")
    except ImportError:
        logger.warning("Could not import ExperimentAnalysis for parallel coordinates plot")

def plot_trial_performance(analysis, logger, plot_dir=None, metric="eval_recall", file_name="trials_comparison.png"):
    """
    Plot performance across different trials.

    Args:
        analysis: Ray Tune analysis object
        logger: Logger instance
        plot_dir: Directory for plots (if None, uses config)
        metric: Metric to visualize
        file_name: Output filename
    """

    if plot_dir is None:
        config = get_config()
        plot_dir = config.get("plots_dir")

    # Ensure plot_dir is a Path object
    plot_dir = Path(plot_dir)
    hyperparams_dir = plot_dir / "hyperparams"
    hyperparams_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment data
    df = analysis.dataframe()
    logger.info(f"analysis dataframe : {df.head()}")

    if "trial_id" in df.columns and metric in df.columns:
        # Get final results for each trial
        trial_final = df.groupby("trial_id")[metric].last().reset_index()
        trial_final = trial_final.sort_values("trial_id")

        fig, ax = create_figure(figsize='wide')
        ax.plot(range(len(trial_final)), trial_final[metric], "o",
                alpha=PLOT_PARAMS['alpha'], color=PRIMARY_COLORS['blue'],
                markersize=PLOT_PARAMS['markersize'])
        format_axis(ax,
                    xlabel="Trial Number",
                    ylabel=metric,
                    title=f"Final {metric} Score by Trial",
                    grid=True,
                    grid_axis='y',
                    legend=False)
        save_figure(fig, hyperparams_dir / file_name)
        plt.close(fig)
        logger.info(f"Trial comparison plot saved")

def clear_cuda_cache():
    """Clear CUDA cache and log memory usage."""
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Cleared CUDA cache. Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
"""
def mail_report(message,subject='Classifier report'):



    msg = MIMEText(message)

    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = you

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    s = smtplib.SMTP('localhost')
    s.sendmail(me, [you], msg.as_string())
    s.quit()
    s = smtplib.SMTP('smtp.gmail.com', 587)
    # start TLS for security
    s.starttls()
    # sending the mail
    s.sendmail("leandrecatogni", "leandrecatogni", message)
    # terminating the session
    s.quit()
    return None

"""