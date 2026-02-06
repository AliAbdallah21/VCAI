"""
Evaluation Module for VCAI

This module provides comprehensive evaluation utilities for assessing
model performance across various metrics and tasks.
"""

from typing import Dict, List, Optional, Any, Union
import importlib
import warnings

__version__ = "1.0.0"
__author__ = "Mena Khaled"

# Module-level exports
__all__ = [
    # Metrics
    "calculate_metrics",
    "compute_accuracy",
    "compute_precision",
    "compute_recall",
    "compute_f1_score",
    "compute_confusion_matrix",
    
    # Evaluators
    "ModelEvaluator",
    "TaskEvaluator",
    
    # Utilities
    "save_evaluation_results",
    "load_evaluation_results",
    "plot_evaluation_metrics",
]


# ============================================================================
# Lazy Imports - Load components only when needed
# ============================================================================

def _lazy_import(module_name: str, attribute: str = None):
    """
    Lazy import helper to defer imports until actually needed.
    
    Args:
        module_name: Name of the module to import
        attribute: Specific attribute to import from module
        
    Returns:
        Imported module or attribute
    """
    try:
        module = importlib.import_module(f".{module_name}", package="evaluation")
        if attribute:
            return getattr(module, attribute)
        return module
    except ImportError as e:
        warnings.warn(
            f"Could not import {module_name} from evaluation module: {e}",
            ImportWarning
        )
        return None


# ============================================================================
# Core Evaluation Functions
# ============================================================================

def calculate_metrics(
    predictions: List[Any],
    ground_truth: List[Any],
    metrics: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Calculate specified evaluation metrics.
    
    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels/values
        metrics: List of metric names to calculate. If None, calculates all.
        **kwargs: Additional arguments for specific metrics
        
    Returns:
        Dictionary mapping metric names to values
    """
    metrics_module = _lazy_import("metrics")
    if metrics_module is None:
        raise ImportError("Metrics module not available")
    
    return metrics_module.calculate_metrics(
        predictions, ground_truth, metrics, **kwargs
    )


def compute_accuracy(predictions: List[Any], ground_truth: List[Any]) -> float:
    """Calculate accuracy score."""
    metrics_module = _lazy_import("metrics")
    if metrics_module and hasattr(metrics_module, "compute_accuracy"):
        return metrics_module.compute_accuracy(predictions, ground_truth)
    
    # Fallback implementation
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions) if predictions else 0.0


def compute_precision(
    predictions: List[Any],
    ground_truth: List[Any],
    average: str = "binary"
) -> Union[float, Dict[str, float]]:
    """Calculate precision score."""
    metrics_module = _lazy_import("metrics")
    if metrics_module and hasattr(metrics_module, "compute_precision"):
        return metrics_module.compute_precision(predictions, ground_truth, average)
    
    raise NotImplementedError("Precision calculation requires metrics module")


def compute_recall(
    predictions: List[Any],
    ground_truth: List[Any],
    average: str = "binary"
) -> Union[float, Dict[str, float]]:
    """Calculate recall score."""
    metrics_module = _lazy_import("metrics")
    if metrics_module and hasattr(metrics_module, "compute_recall"):
        return metrics_module.compute_recall(predictions, ground_truth, average)
    
    raise NotImplementedError("Recall calculation requires metrics module")


def compute_f1_score(
    predictions: List[Any],
    ground_truth: List[Any],
    average: str = "binary"
) -> Union[float, Dict[str, float]]:
    """Calculate F1 score."""
    metrics_module = _lazy_import("metrics")
    if metrics_module and hasattr(metrics_module, "compute_f1_score"):
        return metrics_module.compute_f1_score(predictions, ground_truth, average)
    
    raise NotImplementedError("F1 score calculation requires metrics module")


def compute_confusion_matrix(
    predictions: List[Any],
    ground_truth: List[Any]
) -> Any:
    """Calculate confusion matrix."""
    metrics_module = _lazy_import("metrics")
    if metrics_module and hasattr(metrics_module, "compute_confusion_matrix"):
        return metrics_module.compute_confusion_matrix(predictions, ground_truth)
    
    raise NotImplementedError("Confusion matrix requires metrics module")


# ============================================================================
# Evaluator Classes
# ============================================================================

class ModelEvaluator:
    """
    Main evaluator class for model performance assessment.
    
    Example:
        >>> evaluator = ModelEvaluator(model)
        >>> results = evaluator.evaluate(test_data)
        >>> print(results['accuracy'])
    """
    
    def __init__(self, model=None, config: Optional[Dict] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            config: Configuration dictionary
        """
        self._evaluator_impl = None
        self.model = model
        self.config = config or {}
        
    def _get_evaluator(self):
        """Lazy load the evaluator implementation."""
        if self._evaluator_impl is None:
            evaluator_class = _lazy_import("evaluators", "ModelEvaluator")
            if evaluator_class:
                self._evaluator_impl = evaluator_class(self.model, self.config)
            else:
                raise ImportError("ModelEvaluator implementation not found")
        return self._evaluator_impl
    
    def evaluate(self, data, **kwargs) -> Dict[str, Any]:
        """Evaluate model on provided data."""
        return self._get_evaluator().evaluate(data, **kwargs)
    
    def batch_evaluate(self, batches, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on multiple batches."""
        return self._get_evaluator().batch_evaluate(batches, **kwargs)


class TaskEvaluator:
    """
    Task-specific evaluator for specialized evaluation scenarios.
    
    Example:
        >>> evaluator = TaskEvaluator(task_type='classification')
        >>> results = evaluator.run(predictions, labels)
    """
    
    def __init__(self, task_type: str, config: Optional[Dict] = None):
        """
        Initialize task evaluator.
        
        Args:
            task_type: Type of task (e.g., 'classification', 'regression')
            config: Configuration dictionary
        """
        self._evaluator_impl = None
        self.task_type = task_type
        self.config = config or {}
    
    def _get_evaluator(self):
        """Lazy load the task evaluator implementation."""
        if self._evaluator_impl is None:
            evaluator_class = _lazy_import("evaluators", "TaskEvaluator")
            if evaluator_class:
                self._evaluator_impl = evaluator_class(self.task_type, self.config)
            else:
                raise ImportError("TaskEvaluator implementation not found")
        return self._evaluator_impl
    
    def run(self, predictions, ground_truth, **kwargs) -> Dict[str, Any]:
        """Run task-specific evaluation."""
        return self._get_evaluator().run(predictions, ground_truth, **kwargs)


# ============================================================================
# Utility Functions
# ============================================================================

def save_evaluation_results(
    results: Dict[str, Any],
    filepath: str,
    format: str = "json"
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: Dictionary of evaluation results
        filepath: Path to save file
        format: Output format ('json', 'yaml', 'csv')
    """
    utils_module = _lazy_import("utils")
    if utils_module and hasattr(utils_module, "save_results"):
        utils_module.save_results(results, filepath, format)
    else:
        # Fallback to JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


def load_evaluation_results(filepath: str, format: str = "json") -> Dict[str, Any]:
    """
    Load evaluation results from file.
    
    Args:
        filepath: Path to results file
        format: File format ('json', 'yaml', 'csv')
        
    Returns:
        Dictionary of evaluation results
    """
    utils_module = _lazy_import("utils")
    if utils_module and hasattr(utils_module, "load_results"):
        return utils_module.load_results(filepath, format)
    else:
        # Fallback to JSON
        import json
        with open(filepath, 'r') as f:
            return json.load(f)


def plot_evaluation_metrics(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plot evaluation metrics.
    
    Args:
        results: Dictionary of evaluation results
        output_path: Path to save plot (if None, displays plot)
        **kwargs: Additional plotting arguments
    """
    visualization_module = _lazy_import("visualization")
    if visualization_module and hasattr(visualization_module, "plot_metrics"):
        visualization_module.plot_metrics(results, output_path, **kwargs)
    else:
        warnings.warn("Visualization module not available", ImportWarning)


# ============================================================================
# Optional Component Imports
# ============================================================================

def __getattr__(name):
    """
    Dynamic attribute access for lazy loading of submodules.
    
    This allows importing submodules like:
        from evaluation import metrics
        from evaluation import evaluators
    """
    # Map of submodule names
    _submodules = {
        "metrics": "metrics",
        "evaluators": "evaluators",
        "utils": "utils",
        "visualization": "visualization",
        "benchmarks": "benchmarks",
        "datasets": "datasets",
    }
    
    if name in _submodules:
        return _lazy_import(_submodules[name])
    
    raise AttributeError(f"module 'evaluation' has no attribute '{name}'")


# ============================================================================
# Module Initialization
# ============================================================================

def _check_dependencies():
    """Check if required dependencies are available."""
    required = []
    optional = ['matplotlib', 'seaborn', 'plotly']
    
    missing_required = []
    for package in required:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_required.append(package)
    
    if missing_required:
        warnings.warn(
            f"Missing required packages: {', '.join(missing_required)}. "
            f"Some features may not work.",
            ImportWarning
        )
    
    missing_optional = []
    for package in optional:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_optional:
        warnings.warn(
            f"Missing optional packages: {', '.join(missing_optional)}. "
            f"Some visualization features may be limited.",
            ImportWarning
        )


# Run dependency check on import
_check_dependencies()


# ============================================================================
# Module Info
# ============================================================================

def get_version():
    """Return the version of the evaluation module."""
    return __version__


def get_available_metrics():
    """Return list of available metric functions."""
    metrics_module = _lazy_import("metrics")
    if metrics_module and hasattr(metrics_module, "get_available_metrics"):
        return metrics_module.get_available_metrics()
    return ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"]


def get_available_evaluators():
    """Return list of available evaluator classes."""
    return ["ModelEvaluator", "TaskEvaluator"]


# ============================================================================
# Cleanup
# ============================================================================

def __dir__():
    """Define which names are visible for dir() and autocomplete."""
    return __all__ + [
        "__version__",
        "__author__",
        "get_version",
        "get_available_metrics",
        "get_available_evaluators",
    ]

