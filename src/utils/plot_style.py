"""
Unified plotting style configuration for BioMoQA project.

This module provides consistent styling across all plots in the project,
including colors, fonts, figure sizes, and common plotting utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, Any, Union, Literal
from contextlib import contextmanager
from pathlib import Path

# =============================================================================
# COLOR PALETTE
# =============================================================================

# Primary colors for main comparisons
PRIMARY_COLORS = {
    'blue': '#3A86FF',
    'orange': '#FB5607',
    'green': '#06A77D',
    'red': '#D62246',
    'purple': '#A23B72',
    'teal': '#2E86AB',
    'gold': '#F18F01',
    'lime': '#8AC926',
}

# Sequential color palettes for models/categories
MODEL_COLORS = ['#3A86FF', '#8338EC', '#06A77D', '#F18F01', '#FF006E', '#2E86AB']

# Diverging colors for comparisons
COMPARISON_COLORS = {
    'group_a': '#3A86FF',
    'group_b': '#FB5607',
}

# Heatmap colormaps
HEATMAP_CMAP = 'RdYlGn'  # For improvement/comparison heatmaps
SEQUENTIAL_CMAP = 'viridis'  # For sequential data
CONFUSION_CMAP = 'Blues'  # For confusion matrices

# =============================================================================
# FIGURE SIZES (width, height in inches)
# =============================================================================

FIGURE_SIZES = {
    'small': (8, 6),
    'medium': (10, 8),
    'large': (12, 10),
    'wide': (14, 6),
    'extra_wide': (16, 6),
    'square': (10, 10),
    'multi_small': (16, 12),  # For 2x2 or 2x3 subplots
    'multi_large': (20, 12),  # For larger multi-panel figures
}

# =============================================================================
# TYPOGRAPHY
# =============================================================================

FONT_SIZES = {
    'small': 9,
    'normal': 10,
    'medium': 11,
    'large': 12,
    'title': 13,
    'main_title': 14,
}

# =============================================================================
# PLOTTING PARAMETERS
# =============================================================================

PLOT_PARAMS = {
    'dpi': 300,  # High quality for publications
    'linewidth': 2,
    'markersize': 6,
    'alpha': 0.7,
    'grid_alpha': 0.3,
}

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

def setup_plotting_style():
    """
    Set up the default plotting style for the entire project.
    Call this once at the start of your script or notebook.
    """
    # Use seaborn's whitegrid style as base
    sns.set_style("whitegrid")

    # Update matplotlib rcParams
    plt.rcParams.update({
        # Figure
        'figure.dpi': 100,  # Display DPI (save DPI is set separately)
        'figure.facecolor': 'white',
        'savefig.dpi': PLOT_PARAMS['dpi'],
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',

        # Fonts
        'font.size': FONT_SIZES['normal'],
        'font.family': 'sans-serif',
        'axes.titlesize': FONT_SIZES['title'],
        'axes.labelsize': FONT_SIZES['medium'],
        'xtick.labelsize': FONT_SIZES['normal'],
        'ytick.labelsize': FONT_SIZES['normal'],
        'legend.fontsize': FONT_SIZES['normal'],

        # Lines and markers
        'lines.linewidth': PLOT_PARAMS['linewidth'],
        'lines.markersize': PLOT_PARAMS['markersize'],

        # Grid
        'grid.alpha': PLOT_PARAMS['grid_alpha'],
        'grid.linestyle': '--',

        # Axes
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.labelweight': 'normal',
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


@contextmanager
def plot_context(style: str = 'default', figsize: Optional[Tuple[float, float]] = None):
    """
    Context manager for temporary plotting style changes.

    Args:
        style: Style preset ('paper', 'presentation', 'notebook')
        figsize: Figure size override

    Example:
        with plot_context('paper', figsize=(10, 8)):
            plt.plot(x, y)
            plt.savefig('plot.png')
    """
    # Save current settings
    old_params = plt.rcParams.copy()

    # Apply context-specific settings
    if style == 'paper':
        plt.rcParams.update({
            'font.size': FONT_SIZES['normal'],
            'axes.titlesize': FONT_SIZES['large'],
            'axes.labelsize': FONT_SIZES['medium'],
        })
    elif style == 'presentation':
        plt.rcParams.update({
            'font.size': FONT_SIZES['medium'],
            'axes.titlesize': FONT_SIZES['main_title'],
            'axes.labelsize': FONT_SIZES['large'],
            'lines.linewidth': 2.5,
        })
    elif style == 'notebook':
        plt.rcParams.update({
            'font.size': FONT_SIZES['normal'],
        })

    try:
        yield
    finally:
        # Restore original settings
        plt.rcParams.update(old_params)


def get_color_palette(n_colors: int, palette_type: str = 'model') -> list:
    """
    Get a color palette with n colors.

    Args:
        n_colors: Number of colors needed
        palette_type: Type of palette ('model', 'sequential', 'diverging')

    Returns:
        List of color hex codes
    """
    if palette_type == 'model':
        if n_colors <= len(MODEL_COLORS):
            return MODEL_COLORS[:n_colors]
        else:
            # Use seaborn to generate more colors if needed
            return sns.color_palette('husl', n_colors).as_hex()
    elif palette_type == 'sequential':
        return sns.color_palette(SEQUENTIAL_CMAP, n_colors).as_hex()
    elif palette_type == 'diverging':
        return sns.color_palette('RdBu_r', n_colors).as_hex()
    else:
        return sns.color_palette('Set2', n_colors).as_hex()


def save_figure(fig, filepath: Union[str, Path], **kwargs):
    """
    Save a figure with consistent settings.

    Args:
        fig: Matplotlib figure object
        filepath: Path to save the figure (str or Path object)
        **kwargs: Additional arguments to pass to fig.savefig()
    """
    default_kwargs = {
        'dpi': PLOT_PARAMS['dpi'],
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none',
    }
    default_kwargs.update(kwargs)
    fig.savefig(str(filepath), **default_kwargs)


def format_axis(ax,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                title: Optional[str] = None,
                grid: bool = True,
                grid_axis: str = 'both',
                legend: bool = True,
                legend_kwargs: Optional[Dict[str, Any]] = None):
    """
    Apply consistent formatting to an axis.

    Args:
        ax: Matplotlib axis object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        grid: Whether to show grid
        grid_axis: Which axis to show grid on ('x', 'y', 'both')
        legend: Whether to show legend
        legend_kwargs: Additional arguments for legend
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES['medium'], fontweight='normal')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES['medium'], fontweight='normal')
    if title:
        ax.set_title(title, fontsize=FONT_SIZES['title'], fontweight='bold', pad=10)

    if grid:
        ax.grid(True, axis=grid_axis, alpha=PLOT_PARAMS['grid_alpha'], linestyle='--')

    if legend and ax.get_legend_handles_labels()[0]:
        default_legend_kwargs = {'loc': 'best', 'frameon': True, 'framealpha': 0.9}
        if legend_kwargs:
            default_legend_kwargs.update(legend_kwargs)
        ax.legend(**default_legend_kwargs)


FigSizeType = Union[
    Tuple[float, float],
    Literal['small', 'medium', 'large', 'wide', 'extra_wide', 'square', 'multi_small', 'multi_large']
]

def create_figure(figsize: Optional[FigSizeType] = None,
                  nrows: int = 1,
                  ncols: int = 1,
                  **kwargs) -> Tuple:
    """
    Create a figure with consistent settings.

    Args:
        figsize: Figure size as (width, height) tuple or a size key from FIGURE_SIZES
                 ('small', 'medium', 'large', 'wide', 'extra_wide', 'square',
                  'multi_small', 'multi_large')
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        **kwargs: Additional arguments for plt.subplots()

    Returns:
        Figure and axes objects
    """
    # Handle figsize
    actual_figsize: Tuple[float, float]

    if figsize is None:
        if nrows == 1 and ncols == 1:
            actual_figsize = FIGURE_SIZES['medium']
        elif nrows * ncols <= 4:
            actual_figsize = FIGURE_SIZES['multi_small']
        else:
            actual_figsize = FIGURE_SIZES['multi_large']
    elif isinstance(figsize, str):
        actual_figsize = FIGURE_SIZES.get(figsize, FIGURE_SIZES['medium'])
    else:
        actual_figsize = figsize

    fig, axes = plt.subplots(nrows, ncols, figsize=actual_figsize, **kwargs)
    return fig, axes


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON PLOT TYPES
# =============================================================================

def style_roc_curve(ax):
    """Apply consistent styling to ROC curve plots."""
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--',
            alpha=0.6, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    format_axis(ax,
                xlabel='False Positive Rate',
                ylabel='True Positive Rate',
                title='ROC Curve',
                grid=True,
                grid_axis='both')


def style_pr_curve(ax):
    """Apply consistent styling to Precision-Recall curve plots."""
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    format_axis(ax,
                xlabel='Recall',
                ylabel='Precision',
                title='Precision-Recall Curve',
                grid=True,
                grid_axis='both')


def style_confusion_matrix(ax):
    """Apply consistent styling to confusion matrix plots."""
    ax.set_xlabel('Predicted Label', fontsize=FONT_SIZES['medium'])
    ax.set_ylabel('True Label', fontsize=FONT_SIZES['medium'])
    ax.set_title('Confusion Matrix', fontsize=FONT_SIZES['title'], fontweight='bold')


def abbreviate_model_name(model_name: str) -> str:
    """
    Abbreviate long model names for cleaner plot labels.

    Args:
        model_name: Full model name

    Returns:
        Abbreviated model name
    """
    abbreviations = {
        'BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext': 'BiomedBERT-AF',
        'BiomedNLP-BiomedBERT-base-uncased-abstract': 'BiomedBERT-A',
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext': 'BiomedBERT-AF',
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract': 'BiomedBERT-A',
        'FacebookAI/roberta-base': 'RoBERTa',
        'google-bert/bert-base-uncased': 'BERT',
        'dmis-lab/biobert-v1.1': 'BioBERT',
    }

    for full, abbrev in abbreviations.items():
        if full in model_name:
            model_name = model_name.replace(full, abbrev)

    # Remove common prefixes
    model_name = model_name.replace('BiomedNLP-', '')

    return model_name


# Initialize style on import
setup_plotting_style()
