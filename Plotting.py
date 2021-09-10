import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import seaborn as sns
import numpy as np

from typing import Tuple, List, NewType
from functools import reduce

Figure = NewType('Figure', matplotlib.figure.Figure)
Axis   = NewType('Figure', matplotlib.axes.Axes)
l_multiply = lambda x, y: x * y
l_sum      = lambda x, y: x + y

def stacked_frames(num_rows: int, num_cols: int, size: Tuple[int],
                   names_left: List[str] = None,
                   names_right: List[str] = None,
                   x_axis_name: str = None,
                   title: str = None,
                   names_size: int = 15,
                   title_size: int = 20)         -> Tuple[Figure, Axis]:
    fig, axs = plt.subplots(nrows = num_rows, ncols = num_cols)
    fig.set_size_inches(size)
    fig.subplots_adjust(hspace = 0.01)

    # remove the x - axis from all frames, minus the bottom one
    for ax in axs[:-1]:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        
    # remove the y - axis from all frames
    for ax in axs:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.set_yticks([])
        
    # limiting the number of ticks in the bottom axis
    axs[-1].locator_params('x', nbins = 4)
    
    # setting names at the left side of each frame
    if names_left != None and len(names_left) == len(axs):
        for ax, name in zip(axs, names_left):
            ax.text(-.03, 0.5, name, fontsize = names_size, rotation = "vertical", 
                    transform = ax.transAxes, va = 'center', family = 'serif')
            
    if names_right != None and len(names_right) == len(axs):
        for ax, name in zip(axs, names_right):
            ax.text(1.01, 0.5, name, fontsize = names_size, rotation = "vertical",
                    transform = ax.transAxes, va = 'center', family = 'serif')
    
    # setting name at the bottom of the last frame
    if x_axis_name != None:
        axs[-1].set_xlabel(x_axis_name, fontsize = names_size, family = 'serif')    
        
    # setting title
    if title != None:
        axs[0].set_title(title, fontsize = 20, family = 'serif')
        
    return fig, axs

def grid_frames(num_rows: int, num_cols: int, size: Tuple[int] = None,
                spacing: Tuple[float] = None,
                remove_all_axis: bool = False,
                x_names: List[str] = None,
                y_names: List[str] = None,
                axs_titles: List[str] = None,
                title: str = None,
                names_size: int = 12,
                title_size: int = 18) -> Tuple[Figure, Axis]:
    fig, axs = plt.subplots(nrows = num_rows, ncols = num_cols)
    if size == None: 
        fig.set_size_inches(3*num_cols, 2.5*num_rows)
    else: 
        fig.set_size_inches(size)

    if spacing == None:
        fig.subplots_adjust(hspace = max(0.35, 0.15*num_rows),
                            wspace = max(0.5, 0.07*num_cols))
    else:
        fig.subplots_adjust(spacing)
    
    # Remove or format the axis's ticks
    if remove_all_axis:
        # remove the y - axis from all frames
        for ax in axs.flatten():
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.set_yticks([])
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.set_xticks([])
    else:
        # limiting the number of ticks in the bottom axis
        for ax in axs.flatten():
            ax.locator_params('x', nbins = 4)
            ax.locator_params('y', nbins = 4)
            ax.xaxis.set_tick_params(labelsize = 7)
            ax.yaxis.set_tick_params(labelsize = 7)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%g'))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%g'))
        
    # Set x labels
    if x_names != None and len(x_names) == reduce(l_multiply, axs.shape):
        for ax, name in zip(axs.flatten(), x_names):
            ax.set_xlabel(name, fontsize = names_size, family = 'serif') 
        
    # Set y labels
    if y_names != None and len(y_names) == reduce(l_multiply, axs.shape):
        for ax, name in zip(axs.flatten(), y_names):
            ax.set_ylabel(name, fontsize = names_size, family = 'serif')
    
    # Set title
    if title != None:
        fig.suptitle(title, fontsize= title_size)
    
    return fig, axs

# def heat_plot(X: np.array, size: Tuple[int],
#               x_tick_label_size: int,
#               y_tick_label_size: int,
#               title: str,
#               title_size: int,
#               x_tick_labels: List[str] = None,
#               y_tick_labels: List[str] = None,
#               y_tick_rotation: str = 'horizontal',
#               cmap: str = None) -> Tuple[Figure, Axis]:
    
#     sns.heatmap(X, linewidth = 0, ax = ax, cmap = cmap)
    
#     ax.locator_params('x', nbins = 3)
#     ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    
#     if x_tick_labels != None:
#         ax.set_xticklabels(x_tick_labels)
#     if y_tick_labels != None:
#         ax.set_yticklabels(y_tick_labels, rotation = y_tick_rotation)
        
#     ax.tick_params(axis = 'x', which = 'both', labelsize = x_tick_label_size)
#     ax.tick_params(axis = 'y', which = 'both', labelsize = y_tick_label_size)
    
#     ax.set_title(title, size = title_size)
    
#     return fig, ax

def simplePlot(size: Tuple[int, int], x_label: str, y_label: str, title: str,
               x_label_size: int, y_label_size: int, title_size: int,
               x_tick_label_size: int, y_tick_label_size: int) -> Tuple[Figure, Axis]:
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(size)
    
    ax.set_xlabel(x_label, size = x_label_size)
    ax.set_ylabel(y_label, size = y_label_size)
    ax.set_title(title, size = title_size)
    
    ax.tick_params(axis='x', which='both', labelsize= x_tick_label_size)
    ax.tick_params(axis='y', which='both', labelsize= y_tick_label_size)
    
    return fig, ax