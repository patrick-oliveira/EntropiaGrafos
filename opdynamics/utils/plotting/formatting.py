import numpy as np

from opdynamics.utils.types import Figure, Axis
from typing import Tuple, Union, List
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

def classical_structure(
    fig: Figure, 
    axis: Union[Axis, np.ndarray],
    axis_width: float = 1.08,
    tick_pad: int = 8,
    tick_label_size: int = 17,
    num_x_ticks: int = 3,
    num_y_ticks: int = 3,
    *arg,
    **kwargs
) -> Tuple[Figure, Axis]:
    if type(axis) != np.ndarray:
        axis = np.array([axis])
        
    for ax in axis.flatten():
        for axis_line in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis_line].set_linewidth(axis_width)
        
        ax.xaxis.set_major_locator(MaxNLocator(num_x_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(num_y_ticks))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax.tick_params(
            which = 'both',
            top = True,
            right = True,
            direction = 'in',
            width = axis_width
        )
        ax.tick_params(
            which = 'major',
            pad = tick_pad,
            labelsize = tick_label_size,
            length = 6
        )
        ax.tick_params(
            which = 'minor',
            length = 3
        )
        
        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label2.set_visible(False)
        xticks[0].tick2line.set_visible(False)
        
        yticks = ax.yaxis.get_major_ticks()
        yticks[0].label2.set_visible(False)
        yticks[0].tick2line.set_visible(False)
        yticks[-1].tick2line.set_visible(False)
         
    return fig, axis

def format_text(
    fig: Figure,
    axis: Union[Axis, np.ndarray],
    titles: List[str],
    title_size: int = 15,
    single_ylabel: bool = True,
    position_ylabel: str = 'left',
    ylabel: str = None,
    ylabel_fontsize: int = 15,
    xlabel_bottom_only: bool = True,
    xlabel: str = None,
    xlabel_fontsize: int = 15,
    *args,
    **kwargs
) -> Tuple[Figure, Union[Axis, np.ndarray]]:
    if type(axis) != np.ndarray:
        axis = np.array([axis])
        
    for ax, title in zip(axis.flatten(), titles):
        ax.set_title(title, y = 1.015)

    if xlabel_bottom_only and len(axis.shape) > 1:
        axs_bottom_row = axis[-1]
        for ax in axs_bottom_row:
            ax.set_xlabel(xlabel)
    else:
        for ax in axis.flatten():
            ax.set_xlabel(xlabel)

    if single_ylabel:
        if len(axis.shape) == 1:
            axis[0].set_ylabel(ylabel)
        else:
            for ax_row in axis:
                ax_row[0].set_ylabel(ylabel)
    else:
        for ax in axis.flatten():
            ax.set_ylabel(ylabel)
            
    for ax in axis.flatten():
        ax.xaxis.label.set_size(xlabel_fontsize)
        ax.yaxis.label.set_size(ylabel_fontsize)
        ax.title.set_size(title_size)
        
    return fig, axis