import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from typing import Tuple, List, NewType
from functools import reduce

Figure = NewType('Figure', matplotlib.figure.Figure)
Axis   = NewType('Axis', matplotlib.axes.Axes)

l_multiply = lambda x, y: x * y
l_sum      = lambda x, y: x + y

def stacked_frames(num_rows: int, num_cols: int, size: Tuple[int],
                   names_left: List[str] = None,
                   names_right: List[str] = None,
                   x_axis_name: str = None,
                   title: str = None,
                   names_size: int = 15,
                   title_size: int = 20)         -> Tuple[Figure, Axis]:
    """
    Build a frame of plots stacked on top of each other, preserving the axis information (ticks are not removed, scale isn't edited, etc.).

    Args:
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.
        size (Tuple[int]): Size of the figure.
        names_left (List[str], optional): Names at the left of each row (not working yet). Defaults to None.
        names_right (List[str], optional): Names at the right of each row (not working yet). Defaults to None.
        x_axis_name (str, optional): A label for the x axis (identifying the x-axis of all plots). Defaults to None.
        title (str, optional): Title of the figure. Defaults to None.
        names_size (int, optional): Font size. Defaults to 15.
        title_size (int, optional): Title size. Defaults to 20.

    Returns:
        Tuple[Figure, Axis]: An instance of Figure and an array of Axis objects.
    """
    assert num_rows > 1, 'Number of rows must be >= 2.'
    
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
        axs[0].set_title(title, fontsize = title_size, family = 'serif')
        
    return fig, axs

def grid_frames(num_rows: int, num_cols: int, size: Tuple[int] = None, # Adicionar títulos para as colunas e linhas
                spacing: Tuple[float] = None,
                remove_all_axis: bool = False,
                x_names: List[str] = None,                                 
                y_names: List[str] = None,
                axs_titles: List[str] = None,
                title: str = None,
                names_size: int = 12,
                ax_title_size: int = 15,
                title_size: int = 18) -> Tuple[Figure, Axis]:
    """
    Build a grid of plots removing or simplifying information on the axis, in order to emphasize the images.

    Args:
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.
        size (Tuple[int], optional): Figure's size.. Defaults to None.
        remove_all_axis (bool, optional): Decides wether or not the x and y axis should be removed. Defaults to False.
        x_names (List[str], optional): A list of label for the x-axis of each plot. Defaults to None.
        y_names (List[str], optional): A list of label for the y-axis of each plot. Defaults to None.
        axs_titles (List[str], optional): A list of titles for each plot. Defaults to None.
        title (str, optional): The figure's title. Defaults to None.
        names_size (int, optional): Font size. Defaults to 12.
        ax_title_size (int, optional): Title size for each plot. Defaults to 15.
        title_size (int, optional): Title size for the whole picture. Defaults to 18.

    Returns:
        Tuple[Figure, Axis]: An instance of Figure and an array of Axis objects.
    """    
    assert num_rows > 1, 'Number of rows must be >= 2.'
    assert num_cols > 1, 'Number of cols must be >= 2.'
    
    fig, axs = plt.subplots(nrows = num_rows, ncols = num_cols)
    if size == None: 
        fig.set_size_inches(3*num_cols, 2.5*num_rows)
    else: 
        fig.set_size_inches(size)

    if spacing == None:
        fig.subplots_adjust(hspace = max(0.35, 0.15*num_rows),
                            wspace = max(0.5, 0.07*num_cols))
    else:
        fig.subplots_adjust(hspace = spacing[0], wspace = spacing[1])
    
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
    
    # Set axis titles
    if axs_titles != None:
        for ax, ax_title in zip(axs.flatten(), axs_titles):
            ax.set_title(ax_title, fontsize = ax_title_size, family = 'serif')
        
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
        fig.suptitle(title, fontsize = title_size)
    
    return fig, axs

def simple_plot(Y, X, size: Tuple[int, int], *args, **kwargs) -> Tuple[Figure, Axis]: # Mudar isso
    fig, ax = plt.subplots(1, 1, figsize = size)
    if X == None:
        X = np.arange(len(Y))
    ax.plot(X, Y, *args, **kwargs)
    
    return fig, ax