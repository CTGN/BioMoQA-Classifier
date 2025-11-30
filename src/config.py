"""
Configuration manager for BioMoQA-Classifier.
Use get_config() to access YAML-driven config for paths, hyperparameters, etc.
Only edit configs/paths.yaml or the Manager's defaults for configuration—never use or define global constants here!
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration manager for BioMoQA-Classifier"""
    
    def __init__(self, project_root: Path = None):
        if project_root is None:
            # Auto-detect project root (should contain pyproject.toml)
            current = Path(__file__).resolve()
            for parent in [current.parent.parent] + list(current.parents):
                if (parent / "pyproject.toml").exists():
                    project_root = parent
                    break
            else:
                project_root = current.parent.parent
                
        self.project_root = project_root
        self.config = self._load_config()
        self._resolve_paths()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files"""
        config_path = self.project_root / "configs" / "paths.yaml"
        
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if YAML file is not available"""
        return {
            "project_root": ".",
            "data_dir": "data",
            "results_dir": "results",
            "plots_dir": "plots",
            "environment": {"seed": 42},
            "training": {"num_folds": 5, "num_runs": 1, "default_optional_negatives": 500}
        }
    
    def _resolve_paths(self):
        """Resolve all path variables and convert to absolute paths"""
        def resolve_string(value: str, context: Dict[str, Any]) -> str:
            """Resolve ${variable} placeholders in strings"""
            import re
            pattern = r'\$\{([^}]+)\}'
            
            def replacer(match):
                var_name = match.group(1)
                return str(context.get(var_name, match.group(0)))
            
            return re.sub(pattern, replacer, value)
        
        def resolve_dict(d: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively resolve paths in dictionary"""
            resolved = {}
            for key, value in d.items():
                if isinstance(value, str):
                    resolved_value = resolve_string(value, context)
                    # Convert to absolute path if it looks like a path
                    if ('/' in resolved_value or '\\' in resolved_value) and not resolved_value.startswith('${'):
                        resolved[key] = self.project_root / resolved_value
                    else:
                        resolved[key] = resolved_value
                elif isinstance(value, dict):
                    resolved[key] = resolve_dict(value, context)
                else:
                    resolved[key] = value
            return resolved
        
        # Create context for variable resolution
        context = {
            "project_root": str(self.project_root),
            "data_dir": "data",
            "results_dir": "results", 
            "plots_dir": "plots",
            "configs_dir": "configs"
        }
        context.update(self.config)
        
        self.config = resolve_dict(self.config, context)
        
        # Ensure critical directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        critical_dirs = [
            self.get("data_dir"),
            self.get("results_dir"), 
            self.get("plots_dir"),
            self.get_path("results", "metrics_dir"),
            self.get_path("results", "final_model_dir"),
            self.get_path("plots", "hyperparams_plots")
        ]
        
        for dir_path in critical_dirs:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                
        logger.info(f"Ensured critical directories exist under {self.project_root}")
    
    def get(self, key: str, default=None):
        """Get a configuration value"""
        return self.config.get(key, default)
    
    def get_path(self, section: str, key: str, default=None) -> Path:
        """Get a path from a specific section"""
        section_config = self.config.get(section, {})
        if isinstance(section_config, dict):
            path_value = section_config.get(key, default)
            return Path(path_value) if path_value else None
        return default
    
    def get_data_path(self, filename: str) -> Path:
        """Get path for data file"""
        return Path(self.get("data_dir")) / filename
    
    def get_results_path(self, filename: str) -> Path:
        """Get path for results file"""
        return Path(self.get("results_dir")) / filename
    
    def get_fold_path(self, split: str, fold: int, run: int) -> Path:
        """Get path for cross-validation fold data"""
        folds_dir = self.get_path("data", "folds_dir")
        if folds_dir:
            return folds_dir / f"{split}{fold}_run-{run}.csv"
        return self.get_data_path(f"biomoqa/folds/{split}{fold}_run-{run}.csv")
    
    def get_model_checkpoint_path(self, loss_type: str, model_name: str, fold: int, nb_opt_negs: int = 0) -> Path:
        """Get path for model checkpoint"""
        from .utils.utils import map_name
        final_model_dir = self.get_path("results", "final_model_dir")
        if final_model_dir:
            checkpoint_name = f"best_model_cross_val_{loss_type}_{map_name(model_name)}_fold-{fold}_opt_negs-{nb_opt_negs}"
            return final_model_dir / checkpoint_name
        return self.get_results_path(f"final_model/best_model_cross_val_{loss_type}_{map_name(model_name)}_fold-{fold}_opt_negs-{nb_opt_negs}")


# Global configuration instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def reload_config(project_root: Path = None):
    """Reload configuration (useful for testing)"""
    global _config_manager
    _config_manager = ConfigManager(project_root)
    return _config_manager
