import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import logging
import json
import argparse
import sys

logger = logging.getLogger(__name__)

class BioMoQAEnsemblePredictor:
    """
    Loads and applies 5-fold cross-validated model ensemble for scoring.
    Predicts ensemble scores for single or batch inputs with optional title.
    """
    def __init__(
        self,
        model_type: str,
        loss_type: str,
        base_path: str = "results/final_model",
        threshold: float = 0.5,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        use_compile: Optional[bool] = None,
    ):
        self.model_type = model_type
        self.loss_type = loss_type
        self.base_path = base_path
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 if use_fp16 is not None else (self.device == "cuda")
        self.use_compile = use_compile if use_compile is not None else False
        self.fold_models = {}
        self.fold_tokenizers = {}
        self.num_folds = 5
        self._load_fold_models()

    def _load_fold_models(self):
        """
        Load all 5 fold models for the specified model type and loss type (required). Fails if any missing.
        """
        for fold in range(1, self.num_folds + 1):
            fold_path = os.path.join(self.base_path, f"best_model_cross_val_{self.loss_type}_{self.model_type}_fold-{fold}")
            if not os.path.exists(fold_path):
                raise FileNotFoundError(f"Fold model not found: {fold_path}")
            tokenizer = AutoTokenizer.from_pretrained(fold_path)
            model = AutoModelForSequenceClassification.from_pretrained(fold_path)
            model.to(self.device)
            model.eval()
            if self.device == "cuda":
                if self.use_fp16:
                    model = model.half()
                if self.use_compile and hasattr(torch, 'compile'):
                    try:
                        model = torch.compile(model, mode="default", dynamic=True)
                    except Exception as e:
                        logger.warning(f"Model compilation failed for fold {fold}: {e}")
                        self.use_compile = False
            self.fold_tokenizers[fold] = tokenizer
            self.fold_models[fold] = model
        logger.info(f"Loaded all {self.num_folds} fold models.")

    def score_text(self, abstract: str, title: Optional[str]=None) -> Dict[str, Any]:
        fold_scores = []
        for fold in range(1, self.num_folds + 1):
            tokenizer = self.fold_tokenizers[fold]
            model = self.fold_models[fold]
            if title:
                tokens = tokenizer(title, abstract, truncation=True, max_length=512, padding=True, return_tensors="pt")
            else:
                tokens = tokenizer(abstract, truncation=True, max_length=512, padding=True, return_tensors="pt")
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                if self.use_fp16 and self.device == "cuda":
                    tokens = {k: v.half() if v.dtype == torch.float32 else v for k, v in tokens.items()}
                outputs = model(**tokens)
                logits = outputs.logits
                score = torch.sigmoid(logits).squeeze().float().cpu().item()
                fold_scores.append(score)
        ensemble_score = float(np.mean(fold_scores))
        stats = {
            "mean_score": ensemble_score,
            "std_score": float(np.std(fold_scores)),
            "min_score": float(np.min(fold_scores)),
            "max_score": float(np.max(fold_scores)),
            "positive_folds": sum(1 for s in fold_scores if s > self.threshold),
            "consensus_strength": max(sum(1 for s in fold_scores if s > self.threshold), sum(1 for s in fold_scores if s <= self.threshold))/self.num_folds
        }
        return {
            "ensemble_score": ensemble_score,
            "ensemble_prediction": int(ensemble_score > self.threshold),
            "fold_scores": fold_scores,
            "statistics": stats
        }

    def score_batch(self, batch: List[Dict[str, Union[str, None]]], batch_size: int = 8) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(batch), batch_size):
            minibatch = batch[i:i+batch_size]
            for entry in minibatch:
                results.append(self.score_text(entry.get("abstract", ""), entry.get("title")))
        return results

    def score_batch_optimized(self, batch: List[Dict[str, Union[str, None]]], batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Optimized batched scoring on GPU:
        - Tokenizes a minibatch once per fold, runs a single forward pass per fold
        - Aggregates fold scores per input to produce ensemble metrics
        """
        if not batch:
            return []

        results: List[Dict[str, Any]] = []
        # Process in minibatches for memory control
        for i in range(0, len(batch), batch_size):
            minibatch = batch[i:i + batch_size]

            # Collect per-fold scores for the whole minibatch
            fold_scores_per_item: List[List[float]] = [[] for _ in range(len(minibatch))]

            # Run each fold once on the minibatch
            for fold in range(1, self.num_folds + 1):
                tokenizer = self.fold_tokenizers[fold]
                model = self.fold_models[fold]

                # Prepare tokenized inputs with optional title as pair
                abstracts = [str(entry.get("abstract", "") or "") for entry in minibatch]
                titles = [entry.get("title") for entry in minibatch]

                if any(titles):
                    tokens = tokenizer(
                        titles,
                        abstracts,
                        truncation=True,
                        max_length=512,
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    tokens = tokenizer(
                        abstracts,
                        truncation=True,
                        max_length=512,
                        padding=True,
                        return_tensors="pt",
                    )

                tokens = {k: v.to(self.device, non_blocking=False) for k, v in tokens.items()}

                with torch.inference_mode():
                    # Cast only float32 tensors when on cuda + fp16
                    if self.use_fp16 and self.device == "cuda":
                        tokens = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in tokens.items()}
                    outputs = model(**tokens)
                    logits = outputs.logits
                    # logits shape: [batch, 1] or [batch]
                    scores = torch.sigmoid(logits).view(-1).float().detach().cpu().tolist()

                for idx, s in enumerate(scores):
                    fold_scores_per_item[idx].append(float(s))

            # Aggregate per-item across folds
            for fold_scores in fold_scores_per_item:
                ensemble_score = float(np.mean(fold_scores)) if fold_scores else 0.0
                stats = {
                    "mean_score": ensemble_score,
                    "std_score": float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0,
                    "min_score": float(np.min(fold_scores)) if fold_scores else 0.0,
                    "max_score": float(np.max(fold_scores)) if fold_scores else 0.0,
                    "positive_folds": sum(1 for s in fold_scores if s > self.threshold),
                    "consensus_strength": (
                        max(
                            sum(1 for s in fold_scores if s > self.threshold),
                            sum(1 for s in fold_scores if s <= self.threshold),
                        ) / self.num_folds
                        if fold_scores
                        else 0.0
                    ),
                }
                results.append(
                    {
                        "ensemble_score": ensemble_score,
                        "ensemble_prediction": int(ensemble_score > self.threshold),
                        "fold_scores": fold_scores,
                        "statistics": stats,
                    }
                )

        return results

    def score_batch_ultra_optimized(
        self,
        batch: List[Dict[str, Union[str, None]]],
        base_batch_size: int = 8,
        use_dynamic_batching: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Ultra-optimized path:
        - Optionally groups inputs by approximate length to reduce padding (dynamic batching)
        - Uses the same GPU-batched execution per fold under the hood
        """
        if not batch:
            return []

        if not use_dynamic_batching:
            return self.score_batch_optimized(batch, batch_size=base_batch_size)

        # Create sortable tuples: (length_score, original_index, item)
        def length_of(entry: Dict[str, Union[str, None]]) -> int:
            abstract = str(entry.get("abstract", "") or "")
            title = str(entry.get("title", "") or "")
            return len(abstract) + (len(title) if title else 0)

        indexed = [(length_of(entry), idx, entry) for idx, entry in enumerate(batch)]
        indexed.sort(key=lambda x: x[0])

        # Process in sorted order to minimize padding, then restore original order
        sorted_items = [t[2] for t in indexed]
        optimized_results = self.score_batch_optimized(sorted_items, batch_size=base_batch_size)

        # Map results back to original order
        restored: List[Optional[Dict[str, Any]]] = [None] * len(batch)
        for (_, original_idx, _), res in zip(indexed, optimized_results):
            restored[original_idx] = res

        # All should be filled; if any None, fill with a default
        for i in range(len(restored)):
            if restored[i] is None:
                restored[i] = {
                    "ensemble_score": 0.0,
                    "ensemble_prediction": 0,
                    "fold_scores": [],
                    "statistics": {
                        "mean_score": 0.0,
                        "std_score": 0.0,
                        "min_score": 0.0,
                        "max_score": 0.0,
                        "positive_folds": 0,
                        "consensus_strength": 0.0,
                    },
                }

        return restored

# --- Data loading utilities ---
def load_data(path: str) -> List[Dict[str, Any]]:
    """
    Loads data from a JSON or CSV file into a list of dicts suitable for batch scoring.
    Accepts formats with fields 'abstract', optional 'title', and any others.
    Skips rows with missing or None/empty abstracts and logs a warning with a sample.
    Attempts to normalize CSV columns when possible.
    """
    ext = os.path.splitext(path)[-1].lower()
    loaded, valid = 0, 0
    records = []
    if ext == ".csv":
        df = pd.read_csv(path)
        # Try to find an abstract column, normalize
        abs_col = None
        for col in df.columns:
            if col.lower().strip() == "abstract":
                abs_col = col
                break
        if not abs_col:
            # Try fuzzy match for typical alternatives
            alt_cols = [c for c in df.columns if "abstract" in c.lower()]
            if alt_cols:
                abs_col = alt_cols[0]
                logger.warning(f"No column named 'abstract'; using closest column '{abs_col}'.")
            else:
                raise ValueError(f"CSV must have an 'abstract' column. Available columns: {', '.join(df.columns)}")
        for i, row in df.iterrows():
            loaded += 1
            abst = row.get(abs_col, None)
            if isinstance(abst, float) and np.isnan(abst):
                abst = None
            if abst is not None and isinstance(abst, str) and abst.strip():
                rec = dict(row)
                rec['abstract'] = str(abst)
                records.append(rec)
                valid += 1
            else:
                logger.warning(f"Skipping row {i} (missing/empty abstract): {row}")
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    loaded += 1
                    abst = None
                    if isinstance(item, dict):
                        # Try both 'abstract' and possible alternative
                        for key in item:
                            if key.lower().strip() == "abstract":
                                abst = item[key]
                                break
                        if abst is None:
                            alt_keys = [k for k in item if "abstract" in k.lower()]
                            if alt_keys:
                                abst = item[alt_keys[0]]
                                logger.warning(f"No key named 'abstract' in item {i}; using closest: '{alt_keys[0]}'")
                        if abst is not None and isinstance(abst, str) and abst.strip():
                            rec = dict(item)
                            rec['abstract'] = str(abst)
                            records.append(rec)
                            valid += 1
                        else:
                            logger.warning(f"Skipping JSON dict item {i} (missing/empty abstract): {item}")
                    elif isinstance(item, str) and item.strip():
                        records.append({'abstract': item})
                        valid += 1
                    else:
                        logger.warning(f"Skipping JSON item {i} (unsupported type or empty string): {item}")
            else:
                raise ValueError("JSON root must be an array of dicts or strings.")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    logger.info(f"Loaded {valid} valid / {loaded} total items from: {path}")
    return records

# --- Unified API entrypoint ---
def instantiate_predictor(model_type: str, loss_type: str="BCE", base_path: str="results/final_model", threshold: float=0.5, device: Optional[str]=None) -> BioMoQAEnsemblePredictor:
    """Returns an ensemble predictor ready for batch or single scoring."""
    return BioMoQAEnsemblePredictor(model_type, loss_type, base_path, threshold, device)

def cli():
    parser = argparse.ArgumentParser(
        description="BioMoQA Ensemble Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_type', type=str, required=True, help='Model name/type (e.g. BiomedBERT-abs)')
    parser.add_argument('--loss_type', type=str, default='BCE', help='Loss type (BCE or focal)')
    parser.add_argument('--base_path', type=str, default='results/final_model', help='Directory with fold model checkpoints')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for ensemble prediction')
    parser.add_argument('--device', type=str, default=None, help='Inference device (cpu, cuda, or auto)')
    parser.add_argument('--input_file', type=str, help='Batch input file (CSV/JSON)')
    parser.add_argument('--abstract', type=str, help='Abstract for single prediction')
    parser.add_argument('--title', type=str, help='Title (if relevant; single text only)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for file predictions')
    parser.add_argument('--output_file', type=str, help='Path to save predictions (CSV or JSON)')
    args = parser.parse_args()

    predictor = BioMoQAEnsemblePredictor(
        model_type=args.model_type,
        loss_type=args.loss_type,
        base_path=args.base_path,
        threshold=args.threshold,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.input_file:
        batch = load_data(args.input_file)
        if not batch:
            print("No valid abstracts loaded from file.", file=sys.stderr)
            sys.exit(1)
        results = predictor.score_batch(batch, batch_size=args.batch_size)
        if args.output_file:
            ext = args.output_file.lower().split('.')[-1]
            if ext == "json":
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                import pandas as pd
                pd.DataFrame(results).to_csv(args.output_file, index=False)
            print(f"Results written to {args.output_file}")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
    elif args.abstract:
        result = predictor.score_text(args.abstract, args.title)
        if args.output_file:
            ext = args.output_file.lower().split('.')[-1]
            if ext == "json":
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                import pandas as pd
                pd.DataFrame([result]).to_csv(args.output_file, index=False)
            print(f"Single prediction written to {args.output_file}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("You must provide either --input_file for batch, or --abstract for single prediction.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    cli()
